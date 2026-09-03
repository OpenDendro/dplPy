import io
import os
import json
import zipfile
import contextlib

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl

# LiPD export needs the optional 'pylipd' dependency; skip this whole module if
# it isn't installed (dplpy itself imports fine without it).
pytest.importorskip("pylipd")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _chron(rwl_path="tests/data/rwl/co021.rwl"):
    rwl = _quiet(dpl.readers, rwl_path)
    rwi = _quiet(dpl.detrend, rwl.copy(), fit="Spline", method="ratio")
    crn = _quiet(dpl.chron, rwi, plot=False)
    return rwl, crn


def _read_meta(lpd_path):
    with zipfile.ZipFile(lpd_path) as z:
        name = [n for n in z.namelist() if n.endswith("metadata.jsonld")][0]
        return json.loads(z.read(name))


def test_to_lipd_writes_valid_bagit_container(tmp_path):
    rwl, crn = _chron()
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "co021"), rwl=rwl)
    assert out.endswith(".lpd") and os.path.exists(out)   # .lpd appended
    with zipfile.ZipFile(out) as z:
        names = [n.split("/")[-1] for n in z.namelist() if n.split("/")[-1]]
    for required in ("bagit.txt", "bag-info.txt", "manifest-md5.txt",
                     "tagmanifest-md5.txt", "metadata.jsonld"):
        assert required in names


def test_chronology_is_primary_measurement_named_trsgi(tmp_path):
    rwl, crn = _chron()
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "co021"), rwl=rwl)
    meta = _read_meta(out)
    mts = meta["paleoData"][0]["measurementTable"]
    # first (primary) table holds the chronology as variableName 'trsgi', flagged primary
    prim_cols = {c.get("variableName"): c for c in mts[0]["columns"]}
    assert "trsgi" in prim_cols
    assert prim_cols["trsgi"].get("isPrimary") in (True, "True")
    # raw ring widths live in the SECOND table, not the primary one
    assert len(mts) == 2
    assert any(c.get("variableName") == "ringWidth" for c in mts[1]["columns"])
    assert not any(c.get("variableName") == "ringWidth" for c in mts[0]["columns"])


def test_chronology_values_roundtrip(tmp_path):
    rwl, crn = _chron()
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "co021"), rwl=rwl)
    with zipfile.ZipFile(out) as z:
        pcsv = [n for n in z.namelist() if "measurement1.csv" in n][0]
        arr = pd.read_csv(z.open(pcsv), header=None)
    # column 2 (index 1) is trsgi
    got = arr.iloc[:, 1].to_numpy(dtype=float)
    assert np.nanmax(np.abs(crn["std"].to_numpy() - got)) < 1e-9


def test_get_timeseries_returns_chronology_as_primary(tmp_path):
    from pylipd import LiPD
    rwl, crn = _chron()
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "co021"), rwl=rwl)
    L = LiPD(); _quiet(L.load, out)
    # the chronology is discoverable under its stored name 'trsgi'
    assert "trsgi" in _quiet(L.get_all_variable_names)
    # and it is the PRIMARY timeseries the standard extraction returns
    ts = _quiet(L.get_timeseries, _quiet(L.get_all_dataset_names), to_dataframe=False)
    entries = list(ts.values())[0]
    primary = [e for e in entries if e.get("paleoData_isPrimary")]
    assert len(primary) == 1                       # exactly one primary series
    assert len(entries) >= 3                        # year + chronology + count present


def test_metadata_autofilled_and_geo_written(tmp_path):
    # wa082 has a header, so site metadata is auto-captured and lands in geo
    rwl, crn = _chron("tests/data/rwl/wa082.rwl")
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "wa082"), rwl=rwl)
    meta = _read_meta(out)
    assert meta.get("archiveType") == "Wood"
    coords = meta.get("geo", {}).get("geometry", {}).get("coordinates")
    assert coords and any(float(c) != 0 for c in coords)   # a real coordinate got written


def test_numeric_dataset_name_is_guarded(tmp_path):
    # a purely-numeric dsname breaks pylipd resolution; to_lipd must not emit one
    rwl, crn = _chron()
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "site"), rwl=rwl, dsname="712")
    meta = _read_meta(out)
    import re
    assert not re.fullmatch(r"[0-9.]+", meta["dataSetName"])


def test_chron_only_no_rwl(tmp_path):
    _, crn = _chron()
    out = _quiet(dpl.to_lipd, crn, str(tmp_path / "chrononly"), dsname="chrononly")
    meta = _read_meta(out)
    assert len(meta["paleoData"][0]["measurementTable"]) == 1   # just the chronology


def test_bad_column_raises(tmp_path):
    _, crn = _chron()
    with pytest.raises(ValueError):
        _quiet(dpl.to_lipd, crn, str(tmp_path / "x"), column="not_a_column")


def test_to_lipd_is_exported():
    assert hasattr(dpl, "to_lipd")


# --------------------------------------------------------------------------- #
# Phase 2: multiple chronologies, running stats, publication, provenance
# --------------------------------------------------------------------------- #
def _chron_ars(rwl_path="tests/data/rwl/ca533.rwl"):
    rwl = _quiet(dpl.readers, rwl_path)
    rwi = _quiet(dpl.detrend, rwl.copy(), fit="Spline", method="ratio")
    ars = _quiet(dpl.chron_ars, rwi)
    stab = _quiet(dpl.chron_stabilized, rwi, running_rbar=True)
    return rwl, ars, stab


def test_multiple_chronologies_all(tmp_path):
    rwl, ars, _ = _chron_ars()
    out = _quiet(dpl.to_lipd, ars, str(tmp_path / "ca533"), rwl=rwl, chronologies="all")
    meta = _read_meta(out)
    mts = meta["paleoData"][0]["measurementTable"]
    # 3 chronologies + 1 raw-width table
    assert len(mts) == 4
    trsgi_tables = [t for t in mts if any(c.get("variableName") == "trsgi" for c in t["columns"])]
    assert len(trsgi_tables) == 3
    # exactly one is primary
    prim = [c for t in mts for c in t["columns"]
            if c.get("variableName") == "trsgi" and c.get("isPrimary") in (True, "True")]
    assert len(prim) == 1


def test_running_stats_on_primary(tmp_path):
    rwl, ars, stab = _chron_ars()
    out = _quiet(dpl.to_lipd, ars, str(tmp_path / "ca533"), rwl=rwl,
                 chronologies="all", stats=stab)
    meta = _read_meta(out)
    prim = meta["paleoData"][0]["measurementTable"][0]["columns"]
    names = {c.get("variableName") for c in prim}
    assert "RBAR" in names and "EPS" in names       # running stats added to the primary


def test_publication_and_provenance(tmp_path):
    rwl, ars, _ = _chron_ars()
    out = _quiet(dpl.to_lipd, ars, str(tmp_path / "ca533"), rwl=rwl,
                 provenance="spline detrend; biweight mean",
                 publication={"authors": "Lamarche, V.C.; Graybill, D.A.",
                              "year": 1991, "doi": "10.0/demo"})
    meta = _read_meta(out)
    pubs = meta.get("pub", [])
    assert pubs and str(pubs[0].get("year")) == "1991"
    assert pubs[0].get("doi") == "10.0/demo"
    trsgi = [c for c in meta["paleoData"][0]["measurementTable"][0]["columns"]
             if c.get("variableName") == "trsgi"][0]
    assert "spline detrend" in (trsgi.get("description") or "")
