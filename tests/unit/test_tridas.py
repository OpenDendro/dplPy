# Tests for TRiDaS read/write: dpl.read_tridas / dpl.write_tridas, the
# dpl.readers() single-table delegation, and the dpl.writers(format="tridas")
# convenience path.

import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _meta():
    return dpl.TridasMetadata(
        project_title="Test project", project_types=["dating"],
        category="archaeology", investigator="K. Anchukaitis", period="2026",
        objects=[{"title": "Test site", "type": "forest"}])


def _quiet_write(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dpl.write_tridas(*a, **k)


def _rwl():
    idx = pd.Index(range(1998, 2005), name="Year")
    return pd.DataFrame({
        "ABC01A": [0.50, 0.00, np.nan, 0.52, 0.48, 0.60, 0.55],   # real 0 + gap
        "ABC01B": [0.45, 0.47, 0.49, 0.51, np.nan, np.nan, 0.58],
    }, index=idx)


def _write_xml(tmp_path, text):
    p = tmp_path / "f.xml"
    p.write_text(text)
    return str(p)


# a minimal single-table file in micrometres (0.001 mm steps)
_MICRON_FILE = """<?xml version="1.0"?>
<tridas xmlns="http://www.tridas.org/1.2.2"><project><title>p</title>
<type>dating</type><category>c</category><investigator>i</investigator>
<period>y</period>
<object><title>site</title><type>forest</type>
<element><title>T1</title><taxon>PCGL</taxon><sample><title>c</title><type>core</type>
<radius><title>1</title>
<measurementSeries><title>T1A</title><comments>T1A</comments>
<measuringMethod normalTridas="measuring platform"/>
<interpretation><firstYear suffix="AD">2000</firstYear>
<lastYear suffix="AD">2003</lastYear></interpretation>
<values><variable normalTridas="ring width"/><unit normalTridas="micrometres"/>
<value value="500"/><value/><value value="0"/><value value="620"/>
</values></measurementSeries>
</radius></sample></element></object></project></tridas>"""


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #
def test_roundtrip_values_nan_zero(tmp_path):
    df = _rwl()
    out = _quiet_write(str(tmp_path / "rt"), rwl=df, metadata=_meta(),
                       taxon="PCGL", prec=0.01)
    back = dpl.read_tridas(out).rwl()
    b = back[df.columns].reindex(df.index)
    assert np.nanmax(np.abs(b.to_numpy() - df.to_numpy())) < 1e-9
    # NaN gaps preserved, and a real 0 ring stays 0 (never turned into NaN)
    assert np.array_equal(np.isnan(b.to_numpy()), np.isnan(df.to_numpy()))
    assert back.loc[1999, "ABC01A"] == 0.0


def test_bc_ad_years_roundtrip(tmp_path):
    idx = pd.Index(range(-3, 4), name="Year")       # 4 BC .. 3 AD
    df = pd.DataFrame({"S1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}, index=idx)
    out = _quiet_write(str(tmp_path / "bc"), rwl=df, metadata=_meta(), taxon="PCGL")
    xml = open(out).read()
    assert 'suffix="BC">4' in xml and 'suffix="AD">3' in xml
    back = dpl.read_tridas(out).rwl()
    assert list(back.index) == list(df.index)


def test_kind_tags(tmp_path):
    out = _quiet_write(str(tmp_path / "k"), rwl=_rwl(), metadata=_meta(), taxon="PCGL")
    assert dpl.read_tridas(out).rwl().attrs.get("kind") == "rwl"


# --------------------------------------------------------------------------- #
# reading specifics
# --------------------------------------------------------------------------- #
def test_unit_conversion_to_mm(tmp_path):
    proj = dpl.read_tridas(_write_xml(tmp_path, _MICRON_FILE))
    frame = proj.rwl()
    # 500 micrometres == 0.5 mm; 620 um == 0.62 mm
    assert frame.loc[2000, "T1A"] == pytest.approx(0.500)
    assert frame.loc[2003, "T1A"] == pytest.approx(0.620)
    # precision recorded from the unit
    assert frame.attrs["dplpy_precision"]["T1A"] == pytest.approx(0.001)


def test_missing_value_is_nan_real_zero_kept(tmp_path):
    frame = dpl.read_tridas(_write_xml(tmp_path, _MICRON_FILE)).rwl()
    assert np.isnan(frame.loc[2001, "T1A"])     # the empty <value/>
    assert frame.loc[2002, "T1A"] == 0.0        # value="0" is a real ring


def test_chronology_roundtrip(tmp_path):
    crn = pd.DataFrame({"std": [0.98, 1.02, 1.05, 0.95],
                        "samp_depth": [3, 5, 5, 2]},
                       index=pd.Index(range(1990, 1994), name="Year"))
    out = _quiet_write(str(tmp_path / "crn"), crn=crn, metadata=_meta(),
                       validate=False)
    xml = open(out).read()
    assert "<unitless" in xml and "count=" in xml
    d = dpl.read_tridas(out).derived["std"]
    assert np.allclose(d["std"].to_numpy(), crn["std"].to_numpy())
    assert np.array_equal(d["samp_depth"].to_numpy(),
                          crn["samp_depth"].to_numpy().astype(float))
    assert d.attrs.get("kind") == "crn"


def test_non_width_variable_excluded_and_warns(tmp_path):
    density = _MICRON_FILE.replace('normalTridas="ring width"',
                                   'normalTridas="ring density"')
    with pytest.warns(UserWarning):
        proj = dpl.read_tridas(_write_xml(tmp_path, density))
    assert len(proj.measurements) == 0          # density kept out of rwl tables


def test_duplicate_series_kept_separate_with_warning(tmp_path):
    dup = _MICRON_FILE.replace(
        "</measurementSeries>\n</radius>",
        "</measurementSeries>"
        "<measurementSeries><title>T1A</title><comments>T1A</comments>"
        "<interpretation><firstYear suffix=\"AD\">2000</firstYear>"
        "<lastYear suffix=\"AD\">2001</lastYear></interpretation>"
        "<values><variable normalTridas=\"ring width\"/>"
        "<unit normalTridas=\"micrometres\"/>"
        "<value value=\"510\"/><value value=\"520\"/></values>"
        "</measurementSeries></radius>")
    with pytest.warns(UserWarning):
        frame = dpl.read_tridas(_write_xml(tmp_path, dup)).rwl()
    assert "T1A" in frame.columns and "T1A_2" in frame.columns   # not merged


# --------------------------------------------------------------------------- #
# readers() delegation contract
# --------------------------------------------------------------------------- #
def test_readers_returns_single_table(tmp_path):
    out = _quiet_write(str(tmp_path / "s"), rwl=_rwl(), metadata=_meta(),
                       taxon="PCGL", prec=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = dpl.readers(out)
    assert list(frame.columns) == ["ABC01A", "ABC01B"]
    assert frame.attrs["dplpy_metadata"]["site_name"] == "Test site"


def test_readers_sniffs_tridas_by_content(tmp_path):
    # a TRiDaS file with a non-.xml suffix is detected by content
    p = tmp_path / "mystery.txt"
    p.write_text(_MICRON_FILE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = dpl.readers(str(p))
    assert "T1A" in frame.columns


def test_readers_multitable_raises(tmp_path):
    two = _MICRON_FILE.replace(
        "</object></project>",
        "</object>"
        "<object><title>siteB</title><type>forest</type>"
        "<element><title>T2</title><taxon>QURO</taxon><sample><type>core</type>"
        "<radius><measurementSeries><comments>T2A</comments>"
        "<interpretation><firstYear suffix=\"AD\">2000</firstYear>"
        "<lastYear suffix=\"AD\">2001</lastYear></interpretation>"
        "<values><variable normalTridas=\"ring width\"/>"
        "<unit normalTridas=\"micrometres\"/>"
        "<value value=\"600\"/><value value=\"610\"/></values>"
        "</measurementSeries></radius></sample></element></object></project>")
    path = _write_xml(tmp_path, two)
    with pytest.raises(ValueError, match="read_tridas"):
        dpl.readers(path)
    assert len(dpl.read_tridas(path).measurements) == 2


def test_readers_chronology_file_raises(tmp_path):
    crn = pd.DataFrame({"std": [1.0, 1.1], "samp_depth": [2, 3]},
                       index=pd.Index(range(2000, 2002), name="Year"))
    out = _quiet_write(str(tmp_path / "c"), crn=crn, metadata=_meta(),
                       validate=False)
    with pytest.raises(ValueError, match="read_tridas"):
        dpl.readers(out)


# --------------------------------------------------------------------------- #
# writing / validation
# --------------------------------------------------------------------------- #
def test_mandatory_validation_raises(tmp_path):
    with pytest.raises(ValueError, match="mandatory"):
        dpl.write_tridas(str(tmp_path / "bad"), rwl=_rwl(),
                         metadata=dpl.TridasMetadata(project_title="only"),
                         taxon=None)


def test_validate_false_writes_placeholders(tmp_path):
    out = dpl.write_tridas(str(tmp_path / "ok"), rwl=_rwl(),
                           metadata=dpl.TridasMetadata(project_title="only"),
                           taxon=None, validate=False)
    xml = open(out).read()
    assert "http://www.tridas.org/1.2.2" in xml          # 1.2.2 namespace
    assert "<investigator>unknown</investigator>" in xml  # sanctioned default


def test_writers_format_tridas_delegates(tmp_path):
    import os
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dpl.writers(_rwl(), str(tmp_path / "viaw"), "tridas", header=None)
    assert os.path.exists(str(tmp_path / "viaw.xml"))


def test_ids_grouping(tmp_path):
    # two cores of one tree -> one element with two measurementSeries
    df = pd.DataFrame({
        "ABC01A": [0.5, 0.6, 0.7],
        "ABC01B": [0.4, 0.5, 0.6],
    }, index=pd.Index(range(2000, 2003), name="Year"))
    ids = dpl.read_ids(df)        # tree ABC01, cores A/B
    out = _quiet_write(str(tmp_path / "g"), rwl=df, metadata=_meta(),
                       taxon="PCGL", ids=ids)
    xml = open(out).read()
    assert xml.count("<element>") == 1            # grouped under one tree
    assert xml.count("<measurementSeries>") == 2


# --------------------------------------------------------------------------- #
# v0.9.0 audit follow-ups (C2, C6)
# --------------------------------------------------------------------------- #
def test_disjoint_series_give_contiguous_index(tmp_path):
    # C6: two series 1800-1850 and 1900-1959 -> one frame on the full year range
    # (gap years NaN), not 111 rows.
    a = pd.DataFrame({"A": np.linspace(0.5, 1.0, 51)}, index=pd.Index(range(1800, 1851), name="Year"))
    b = pd.DataFrame({"B": np.linspace(0.5, 1.0, 60)}, index=pd.Index(range(1900, 1960), name="Year"))
    both = pd.concat([a, b], axis=1)                      # 111 rows, non-contiguous
    f = _quiet_write(str(tmp_path / "disjoint"), rwl=both, metadata=_meta(), taxon="PCGL")
    frame = dpl.read_tridas(f).rwl()
    assert list(frame.index) == list(range(1800, 1960))
    assert np.isnan(frame.loc[1875, "A"]) and np.isnan(frame.loc[1875, "B"])
    assert frame.loc[1850, "A"] == pytest.approx(1.0)


def test_unrounded_millimetres_leave_precision_unset(tmp_path):
    # C6: prec=None (and no attr) writes unrounded mm; the reader must not then
    # claim a 1.0 mm precision for values like 0.52.
    f = _quiet_write(str(tmp_path / "mm"), rwl=_rwl(), metadata=_meta(), taxon="PCGL")
    frame = dpl.read_tridas(f).rwl()
    assert "dplpy_precision" not in frame.attrs
    assert frame.loc[2001, "ABC01A"] == pytest.approx(0.52)


def test_write_tridas_uses_read_precision_by_default(tmp_path):
    # C2: a frame read at 0.01 mm is written in 1/100th millimetres unless prec
    # is given explicitly; mixed precisions fall back to unrounded mm with a warning.
    rwl = _rwl()
    rwl.attrs["dplpy_precision"] = {"ABC01A": 0.01, "ABC01B": 0.01}
    f = _quiet_write(str(tmp_path / "p01"), rwl=rwl, metadata=_meta(), taxon="PCGL")
    txt = open(f).read()
    assert 'normalTridas="1/100th millimetres"' in txt
    back = dpl.read_tridas(f).rwl()
    assert set(back.attrs["dplpy_precision"].values()) == {0.01}
    pd.testing.assert_frame_equal(back, rwl, check_names=False)
    # explicit prec wins
    g = _quiet_write(str(tmp_path / "p001"), rwl=rwl, metadata=_meta(), taxon="PCGL", prec=0.001)
    assert 'normalTridas="micrometres"' in open(g).read()
    # mixed -> warn, unrounded mm
    rwl.attrs["dplpy_precision"] = {"ABC01A": 0.01, "ABC01B": 0.001}
    with pytest.warns(UserWarning, match="mixed"):
        h = dpl.write_tridas(str(tmp_path / "mixed"), rwl=rwl, metadata=_meta(), taxon="PCGL")
    assert 'normalTridas="millimetres"' in open(h).read()
