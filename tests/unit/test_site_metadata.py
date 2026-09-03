import io
import os
import tempfile
import contextlib

import pandas as pd
import pytest

import dplpy as dpl
from dplpy.site_metadata import SiteMetadata


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --------------------------------------------------------------------------- #
# construction & key mapping
# --------------------------------------------------------------------------- #
def test_from_metadata_keeps_known_ignores_unknown():
    meta = {"site_id": "CA533", "site_name": "Campito", "country_region": "California",
            "elevation_m": 3400, "latitude": 37.5, "longitude": -118.2,
            "n_header_lines": 3, "header_raw": ["..."]}   # last two are not fields
    sm = SiteMetadata.from_metadata(meta)
    assert sm.site_id == "CA533"
    assert sm.country_region == "California"
    assert sm.latitude == 37.5
    # unknown keys are ignored, not stored
    assert not hasattr(sm, "n_header_lines")


def test_to_crn_header_maps_names_and_omits_none():
    sm = SiteMetadata(site_id="CA533", site_name="Campito", species_code="PILO",
                      country_region="California", species_name="Pinus longaeva",
                      elevation_m=3400, latitude=37.5, longitude=-118.2,
                      investigators="Lamarche")
    h = sm.to_crn_header()
    # the .crn writer's key names differ from the metadata field names
    assert h["state_country"] == "California"      # <- country_region
    assert h["species"] == "Pinus longaeva"        # <- species_name
    assert h["elevation"] == 3400                  # <- elevation_m
    assert h["site_id"] == "CA533"
    # a None field is omitted entirely (so the writer's missing-key check fires)
    sm2 = SiteMetadata(site_id="X")
    assert "state_country" not in sm2.to_crn_header()
    assert sm2.to_crn_header() == {"site_id": "X"}


def test_coerce_accepts_dict_and_instance():
    sm = SiteMetadata(site_id="X")
    assert SiteMetadata.coerce(sm) is sm
    assert SiteMetadata.coerce({"site_id": "Y"}).site_id == "Y"
    with pytest.raises(TypeError):
        SiteMetadata.coerce(42)


def test_from_rwl_reads_dplpy_metadata_attr():
    df = pd.DataFrame({"S": [0.1, 0.2]}, index=pd.Index([1, 2], name="Year"))
    df.attrs["dplpy_metadata"] = {"site_id": "ZZ", "latitude": 10.0}
    sm = SiteMetadata.from_rwl(df)
    assert sm.site_id == "ZZ" and sm.latitude == 10.0
    # a frame with no captured metadata is a clear error
    with pytest.raises(ValueError):
        SiteMetadata.from_rwl(pd.DataFrame({"S": [1]}))


def test_from_rwl_on_real_headered_file():
    rwl = _quiet(dpl.readers, "tests/data/rwl/wa082.rwl")   # has a 3-line header
    sm = SiteMetadata.from_rwl(rwl)
    assert sm.site_id is not None
    assert sm.species_code is not None


# --------------------------------------------------------------------------- #
# writer integration: SiteMetadata path == dict path (no behaviour change)
# --------------------------------------------------------------------------- #
def test_write_crn_sitemetadata_matches_dict_bytes():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    rwi = _quiet(dpl.detrend, rwl.copy(), fit="Spline", method="ratio")
    crn = _quiet(dpl.chron, rwi, plot=False)

    hdr = dict(site_id="CA533", site_name="Campito", species_code="PILO",
               state_country="California", species="Pinus longaeva",
               elevation="3400", latitude="37.5", longitude="-118.2",
               investigators="Lamarche")
    sm = SiteMetadata(site_id="CA533", site_name="Campito", species_code="PILO",
                      country_region="California", species_name="Pinus longaeva",
                      elevation_m="3400", latitude="37.5", longitude="-118.2",
                      investigators="Lamarche")
    d = tempfile.mkdtemp()
    _quiet(dpl.writers, crn, os.path.join(d, "viaDict"), "crn", header=hdr)
    _quiet(dpl.writers, crn, os.path.join(d, "viaSM"), "crn", header=sm)
    with open(os.path.join(d, "viaDict.crn")) as f:
        a = f.read()
    with open(os.path.join(d, "viaSM.crn")) as f:
        b = f.read()
    assert a == b


def test_write_crn_still_rejects_bad_header():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    rwi = _quiet(dpl.detrend, rwl.copy(), fit="Spline", method="ratio")
    crn = _quiet(dpl.chron, rwi, plot=False)
    d = tempfile.mkdtemp()
    # a SiteMetadata missing required fields -> writer's missing-key error
    with pytest.raises(ValueError):
        _quiet(dpl.writers, crn, os.path.join(d, "bad"), "crn",
               header=SiteMetadata(site_id="only"))
    # a non-dict, non-SiteMetadata header is still rejected
    with pytest.raises(ValueError):
        _quiet(dpl.writers, crn, os.path.join(d, "bad2"), "crn", header="nope")


def test_sitemetadata_is_exported():
    assert hasattr(dpl, "SiteMetadata")
