import io
import contextlib

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl

pytest.importorskip("pylipd")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _roundtrip(tmp_path, rwl_path="tests/data/rwl/ca533.rwl"):
    rwl = _quiet(dpl.readers, rwl_path)
    rwi = _quiet(dpl.detrend, rwl.copy(), fit="Spline", method="ratio")
    ars = _quiet(dpl.chron_ars, rwi)
    stab = _quiet(dpl.chron_stabilized, rwi, running_rbar=True)
    out = _quiet(dpl.to_lipd, ars, str(tmp_path / "rt"), rwl=rwl,
                 chronologies="all", stats=stab)
    return rwl, ars, _quiet(dpl.from_lipd, out)


def test_from_lipd_roundtrips_rwl_exactly(tmp_path):
    rwl, _, r = _roundtrip(tmp_path)
    back = r["rwl"]
    assert back is not None and back.shape == rwl.shape
    cols = [c for c in rwl.columns if c in back.columns]
    assert len(cols) == rwl.shape[1]                 # all series recovered by name
    a, b = rwl[cols].align(back[cols], join="inner", axis=0)
    m = a.notna() & b.notna()
    assert np.nanmax((a - b).abs().where(m).values) == 0.0


def test_from_lipd_roundtrips_chronologies(tmp_path):
    _, ars, r = _roundtrip(tmp_path)
    assert set(r["chronologies"]) == {"standard", "residual", "arstan"}
    std = r["chronologies"]["standard"]["trsgi"]
    a, b = ars["std"].align(std, join="inner")
    m = a.notna() & b.notna()
    assert np.nanmax((a - b).abs().where(m).values) < 1e-9
    # headline chronology carries the running stats companions
    assert "trsgi" in r["chronology"].columns
    assert {"RBAR", "EPS"}.issubset(set(r["chronology"].columns))


def test_from_lipd_metadata_shape(tmp_path):
    _, _, r = _roundtrip(tmp_path, "tests/data/rwl/wa082.rwl")   # headered file
    md = r["metadata"]
    # dpl.metadata()-shape keys present
    for key in ("site_id", "site_name", "latitude", "longitude", "first_year",
                "last_year", "investigators"):
        assert key in md
    assert md["first_year"] is not None and md["last_year"] is not None


def test_lipd_metadata_matches_from_lipd(tmp_path):
    rwl = _quiet(dpl.readers, "tests/data/rwl/wa082.rwl")
    rwi = _quiet(dpl.detrend, rwl.copy(), fit="Spline", method="ratio")
    out = _quiet(dpl.to_lipd, _quiet(dpl.chron, rwi, plot=False),
                 str(tmp_path / "w"), rwl=rwl)
    md = _quiet(dpl.lipd_metadata, out)
    assert md["latitude"] == _quiet(dpl.from_lipd, out)["metadata"]["latitude"]


def test_from_lipd_finds_chronology_in_a_model_summary(tmp_path):
    # ITRDB/Nam2k layout: the chronology lives in a model summaryTable, not a
    # measurement table. from_lipd must still find it (proposal section 10a).
    from pylipd import LiPD
    from pylipd.classes.dataset import Dataset
    from pylipd.classes.paleodata import PaleoData
    from pylipd.classes.model import Model
    from pylipd.classes.datatable import DataTable

    sdf = pd.DataFrame({"year": [1000, 1001, 1002],
                        "trsgi": [0.5, 1.0, 1.5], "count": [2, 3, 4]})
    sdf.attrs = {
        "year": {"variableName": "year", "units": "yr AD", "number": 1, "TSid": "m-y"},
        "trsgi": {"variableName": "trsgi", "units": "unitless", "number": 2,
                  "TSid": "m-t", "description": "Standard chronology", "isPrimary": True},
        "count": {"variableName": "sampleCount", "units": "count", "number": 3, "TSid": "m-c"},
    }
    st = DataTable(); st.setFileName("m.paleo1model1summary1.csv")
    st.setMissingValue("NaN"); st.setDataFrame(sdf)
    mdl = Model(); mdl.setSummaryTables([st])
    pdo = PaleoData(); pdo.setName("paleo1"); pdo.setModeledBy([mdl])
    ds = Dataset(); ds.setName("modelsummarytest"); ds.setPaleoData([pdo])
    ds.set_non_standard_property("archiveType", "Wood")
    path = str(tmp_path / "modelsummary.lpd")
    L = LiPD(); L.load_datasets([ds]); _quiet(L.create_lipd, "modelsummarytest", path)

    r = _quiet(dpl.from_lipd, path)
    assert r["chronology"] is not None
    assert "trsgi" in r["chronology"].columns
    got = pd.to_numeric(r["chronology"]["trsgi"]).to_numpy()
    assert np.allclose(np.sort(got), [0.5, 1.0, 1.5])


def test_read_exports_present():
    assert hasattr(dpl, "from_lipd") and hasattr(dpl, "lipd_metadata")
