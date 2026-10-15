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


# ---- v0.9.0 audit follow-ups (A16) -----------------------------------------
import warnings


def test_insert_fix_length_warns_about_dropped_measurement():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dpl.edit_rwl(_series(), year=2003, action="insert", value=99)
    msgs = [str(m.message) for m in w if "dropped the measured ring" in str(m.message)]
    assert len(msgs) == 1 and "2000" in msgs[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dpl.edit_rwl(_series(), year=2003, action="insert", value=99, fix_length=False)
    assert not any("dropped" in str(m.message) for m in w)


def test_dataframe_insert_never_drops_a_measurement():
    # dplR's rwl method: fix_length is not applied; the collection gains a year.
    out = dpl.edit_rwl(_frame(), series="A", year=2003, action="insert", value=99)
    assert list(out.index) == list(range(2000, 2006))
    _check(out["A"], range(2000, 2006), [10, 20, 30, 99, 40, 50])
    assert list(out["B"].dropna()) == [1, 2, 3, 4, 5]
    out2 = dpl.edit_rwl(_frame(), series="A", year=2003, action="insert", value=99,
                        fix_last=False)
    assert list(out2.index) == list(range(2001, 2007))
    _check(out2["A"], range(2001, 2007), [10, 20, 30, 99, 40, 50])
    with pytest.raises(ValueError):
        dpl.edit_rwl(_frame(), series="A", year=2003, action="insert", value=99,
                     fix_length=True)


def test_dataframe_delete_trims_vacated_row():
    # the deleted ring shortens the series by a year; a row left all-NaN at the
    # inner end is trimmed (dplR), while rows other series still cover are kept.
    out = dpl.edit_rwl(_frame(), series="A", year=2003, action="delete")
    assert list(out.index) == list(range(2001, 2006))          # B still covers 2001
    _check(out["A"], range(2001, 2006), [np.nan, 10, 20, 40, 50])
    solo = _frame()[["A"]]
    out = dpl.edit_rwl(solo, series="A", year=2003, action="delete")
    assert list(out.index) == list(range(2002, 2006))          # all-NaN 2001 trimmed
    _check(out["A"], range(2002, 2006), [10, 20, 40, 50])
    out = dpl.edit_rwl(solo, series="A", year=2003, action="delete", fix_last=False)
    assert list(out.index) == list(range(2001, 2005))


def test_series_keeps_leading_and_trailing_nan_rows():
    s = pd.Series([np.nan, np.nan, 10.0, 20, 30, 40, 50, np.nan],
                  index=pd.Index(range(1999, 2007), name="Year"), name="S1")
    out = dpl.edit_rwl(s, year=2003, action="change", value=99)
    assert list(out.index) == list(range(1999, 2007))
    _check(out, range(1999, 2007), [np.nan, np.nan, 10, 20, 99, 40, 50, np.nan])
    out = dpl.edit_rwl(s, year=2003, action="delete", fix_length=False)
    assert list(out.index) == list(range(1999, 2007))          # input NaN rows kept
    _check(out, range(1999, 2007), [np.nan, np.nan, np.nan, 10, 20, 40, 50, np.nan])


def test_year_validation():
    with pytest.raises(ValueError, match="whole calendar year"):
        dpl.edit_rwl(_series(), year=np.nan, action="change", value=1.0)
    with pytest.raises(ValueError, match="whole calendar year"):
        dpl.edit_rwl(_series(), year=2003.7, action="change", value=1.0)
    out = dpl.edit_rwl(_series(), year=2003.0, action="change", value=1.0)   # whole float OK
    assert out.loc[2003] == 1.0


def test_change_rejects_negative_and_nan_values():
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=2003, action="change", value=-1.0)
    with pytest.raises(ValueError):
        dpl.edit_rwl(_series(), year=2003, action="change", value=np.nan)
    assert dpl.edit_rwl(_series(), year=2003, action="CHANGE", value=0.0).loc[2003] == 0.0


def test_edit_keeps_attrs_and_sorts_index():
    df = _frame()
    df.attrs["dplpy_precision"] = 0.01
    out = dpl.edit_rwl(df, series="A", year=2003, action="change", value=1.0)
    assert out.attrs["dplpy_precision"] == 0.01 and out.attrs["kind"] == "rwl"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shuf = dpl.edit_rwl(df.sample(frac=1.0, random_state=0), series="A", year=2003,
                            action="change", value=1.0)
    assert any("not sorted" in str(m.message) for m in w)
    pd.testing.assert_frame_equal(shuf, out)
