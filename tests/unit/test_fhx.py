"""Tests for the FHX2 fire-history layer (dplpy.fhx).

The read/write path is validated against burnr: lgr2.fhx is burnr's own example
file, and the structural facts asserted here (row count, series, year range, and
specific events) match burnr's canonical parse of it (burnr::read_fhx -> the lgr2
dataset). The full (year, series, rec_type) table was confirmed identical to
burnr during development.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl

LGR2 = "tests/data/fhx/lgr2.fhx"

_MINI = """\
Name of site   : test
FHX2 FORMAT
1900 2 3
FB
OA
OR

{. 1900
|| 1901
U| 1902
.} 1903
"""


def test_read_fhx_matches_burnr_structure():
    x = dpl.read_fhx(LGR2)
    assert dpl.is_fhx(x)
    assert len(x) == 2681
    assert len(dpl.fhx_series_names(x)) == 26
    assert dpl.fhx_year_range(x) == (1366, 2012)
    # series order is first-appearance, as in burnr
    assert dpl.fhx_series_names(x)[:4] == ["LGR54", "LGR44", "LGR47", "LGR48"]


def test_read_fhx_specific_events():
    x = dpl.read_fhx(LGR2)
    lgr46 = dpl.fhx_get_series(x, "LGR46")
    assert len(lgr46) == 78
    assert dpl.fhx_year_range(lgr46) == (1705, 1849)
    scars = dpl.fhx_get_events(x)["LGR46"]
    assert list(scars[:3]) == [1773, 1786, 1806]
    # the 1773 scar is a dormant-season fire scar
    row = dpl.fhx_get_year(lgr46, 1773)
    assert str(row["rec_type"].iloc[0]) == "dormant_fs"


def test_read_fhx_parses_codes(tmp_path):
    f = tmp_path / "mini.fhx"
    f.write_text(_MINI)
    x = dpl.read_fhx(str(f))
    got = {(int(r.year), r.series, str(r.rec_type)) for r in x.itertuples(index=False)}
    assert got == {
        (1900, "FOO", "inner_year"),
        (1901, "FOO", "recorder_year"), (1901, "BAR", "recorder_year"),
        (1902, "FOO", "unknown_fs"), (1902, "BAR", "recorder_year"),
        (1903, "BAR", "outer_year"),
    }


def test_write_read_roundtrip(tmp_path):
    x = dpl.read_fhx(LGR2)
    out = tmp_path / "rt.fhx"
    dpl.write_fhx(x, str(out))
    y = dpl.read_fhx(str(out))
    a = x[["year", "series", "rec_type"]].astype(str).sort_values(
        ["series", "year"]).reset_index(drop=True)
    b = y[["year", "series", "rec_type"]].astype(str).sort_values(
        ["series", "year"]).reset_index(drop=True)
    assert a.equals(b)


def test_make_rec_type_validates():
    rt = dpl.make_fhx_rec_type(["unknown_fs", "recorder_year"])
    assert list(rt.categories) == list(
        __import__("dplpy._fhx_vocab", fromlist=["x"]).REC_TYPE_LEVELS)
    with pytest.raises(ValueError):
        dpl.make_fhx_rec_type(["not_a_real_type"])


def test_fhx_constructor_and_is_as_fhx():
    x = dpl.fhx([1900, 1901], ["A", "A"], ["recorder_year", "unknown_fs"])
    assert dpl.is_fhx(x)
    assert x.attrs.get("kind") == "fhx"
    # a plain frame casts via as_fhx
    plain = pd.DataFrame({"year": [1900], "series": ["A"],
                          "rec_type": ["unknown_fs"]})
    assert not dpl.is_fhx(plain)            # rec_type not categorical yet
    assert dpl.is_fhx(dpl.as_fhx(plain))
    with pytest.raises(ValueError):
        dpl.as_fhx(pd.DataFrame({"year": [1900]}))


def test_accessors_get_delete_sort():
    x = dpl.read_fhx(LGR2)
    # fhx_get_year / fhx_get_series
    assert set(dpl.fhx_get_year(x, 1800)["year"]) == {1800}
    assert set(dpl.fhx_get_series(x, "LGR46")["series"].astype(str)) == {"LGR46"}
    # delete by series
    d = dpl.fhx_delete(x, series="LGR46")
    assert "LGR46" not in dpl.fhx_series_names(d)
    assert len(d) == len(x) - 78
    # delete by year
    assert (dpl.fhx_delete(x, year=1800)["year"] == 1800).sum() == 0
    # sort by first year: series ordered by earliest year
    s = dpl.fhx_sort(x, by="first_year")
    firsts = [int(dpl.fhx_get_series(s, n)["year"].min()) for n in dpl.fhx_series_names(s)]
    assert firsts == sorted(firsts)


def test_get_event_years_injury_toggle():
    x = dpl.read_fhx(LGR2)
    scars_only = dpl.fhx_get_events(x, scar_event=True, injury_event=False)
    with_injury = dpl.fhx_get_events(x, scar_event=True, injury_event=True)
    # adding injuries can only add event years
    for s in dpl.fhx_series_names(x):
        assert set(scars_only[s]).issubset(set(with_injury[s]))


def test_violates_canon():
    canon = dpl.fhx([1900], ["A"], ["unknown_fs"])
    assert not dpl.fhx_violates_canon(canon)
    noncanon = dpl.fhx([1900], ["A"], ["latelw_fs"])   # outside the FHX2 canon
    assert dpl.fhx_violates_canon(noncanon)
