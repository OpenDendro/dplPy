"""Tests for the FHX2 fire-history model and I/O (the dpl.fhx namespace).

The read/write path is validated against burnr: lgr2.fhx is burnr's own example
file, and the structural facts asserted here (row count, series, year range, and
specific events) match burnr's canonical parse of it (burnr::read_fhx -> the lgr2
dataset). The full (year, series, rec_type) table was confirmed identical to
burnr during development.
"""
import importlib

import pandas as pd
import pytest

import dplpy as dpl

_model = importlib.import_module("dplpy.fhx_model")  # internal helpers

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


def _series(x):
    return list(x["series"].unique())


def test_read_matches_burnr_structure():
    x = dpl.fhx.read(LGR2)
    assert dpl.fhx.is_fhx(x)
    assert len(x) == 2681
    assert len(_series(x)) == 26
    assert (int(x["year"].min()), int(x["year"].max())) == (1366, 2012)
    # series order is first-appearance, as in burnr
    assert _series(x)[:4] == ["LGR54", "LGR44", "LGR47", "LGR48"]


def test_read_specific_events():
    x = dpl.fhx.read(LGR2)
    lgr46 = x[x["series"] == "LGR46"]
    assert len(lgr46) == 78
    assert (int(lgr46["year"].min()), int(lgr46["year"].max())) == (1705, 1849)
    scars = dpl.fhx.events(x)["LGR46"]
    assert list(scars[:3]) == [1773, 1786, 1806]
    # the 1773 scar is a dormant-season fire scar
    row = lgr46[lgr46["year"] == 1773]
    assert str(row["rec_type"].iloc[0]) == "dormant_fs"


def test_read_parses_codes(tmp_path):
    f = tmp_path / "mini.fhx"
    f.write_text(_MINI)
    x = dpl.fhx.read(str(f))
    got = {(int(r.year), r.series, str(r.rec_type)) for r in x.itertuples(index=False)}
    assert got == {
        (1900, "FOO", "inner_year"),
        (1901, "FOO", "recorder_year"), (1901, "BAR", "recorder_year"),
        (1902, "FOO", "unknown_fs"), (1902, "BAR", "recorder_year"),
        (1903, "BAR", "outer_year"),
    }


def test_write_read_roundtrip(tmp_path):
    x = dpl.fhx.read(LGR2)
    out = tmp_path / "rt.fhx"
    dpl.fhx.write(x, str(out))
    y = dpl.fhx.read(str(out))
    a = x[["year", "series", "rec_type"]].astype(str).sort_values(
        ["series", "year"]).reset_index(drop=True)
    b = y[["year", "series", "rec_type"]].astype(str).sort_values(
        ["series", "year"]).reset_index(drop=True)
    assert a.equals(b)


def test_make_rec_type_validates():
    rt = _model._make_rec_type(["unknown_fs", "recorder_year"])
    assert list(rt.categories) == list(
        __import__("dplpy._fhx_vocab", fromlist=["x"]).REC_TYPE_LEVELS)
    with pytest.raises(ValueError):
        _model._make_rec_type(["not_a_real_type"])


def test_constructor_and_is_as_frame():
    x = dpl.fhx.frame([1900, 1901], ["A", "A"], ["recorder_year", "unknown_fs"])
    assert dpl.fhx.is_fhx(x)
    assert x.attrs.get("kind") == "fhx"
    # a plain frame casts via as_frame
    plain = pd.DataFrame({"year": [1900], "series": ["A"],
                          "rec_type": ["unknown_fs"]})
    assert not dpl.fhx.is_fhx(plain)            # rec_type not categorical yet
    assert dpl.fhx.is_fhx(dpl.fhx.as_frame(plain))
    with pytest.raises(ValueError):
        dpl.fhx.as_frame(pd.DataFrame({"year": [1900]}))


def test_sort():
    x = dpl.fhx.read(LGR2)
    s = dpl.fhx.sort(x, by="first_year")
    firsts = [int(s[s["series"] == n]["year"].min()) for n in _series(s)]
    assert firsts == sorted(firsts)


def test_events_injury_toggle():
    x = dpl.fhx.read(LGR2)
    scars_only = dpl.fhx.events(x, scar_event=True, injury_event=False)
    with_injury = dpl.fhx.events(x, scar_event=True, injury_event=True)
    # adding injuries can only add event years
    for s in _series(x):
        assert set(scars_only[s]).issubset(set(with_injury[s]))


def test_violates_canon_internal():
    canon = dpl.fhx.frame([1900], ["A"], ["unknown_fs"])
    assert not _model._violates_canon(canon)
    noncanon = dpl.fhx.frame([1900], ["A"], ["latelw_fs"])   # outside FHX2 canon
    assert _model._violates_canon(noncanon)
