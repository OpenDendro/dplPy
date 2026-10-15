import dplpy as dpl
from dplpy.site_metadata import SiteMetadata
import pandas as pd
import pytest
import io

open_wrapper = io.TextIOWrapper(
    io.BytesIO(),
    encoding='cp1252',
    line_buffering=True,
)
open_wrapper.mode = "w"

def test_write_invalid_type_data():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8],
                                  "Year": [1, 2, 3, 4]})

    with pytest.raises(TypeError) as errorMsg:
        dpl.writers(input_df['SeriesA'], "label", "ext")
    # the exact type repr varies by pandas version (pandas.core.series.Series
    # before pandas 3.0, pandas.Series after), so check the stable part
    msg = str(errorMsg.value)
    assert msg.startswith("Expected dataframe input, got") and "Series" in msg


def test_write_invalid_type_label():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8],
                                  "Year": [1, 2, 3, 4]})

    with pytest.raises(TypeError) as errorMsg:
        dpl.writers(input_df, 1, "ext")
    expected_msg = "Expected label to be of type str, not <class 'int'>"
    assert expected_msg == str(errorMsg.value)


def test_write_invalid_type_format():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8],
                                  "Year": [1, 2, 3, 4]})

    with pytest.raises(TypeError) as errorMsg:
        dpl.writers(input_df, "label", 1)
    expected_msg = "Expected format to be of type str, not <class 'int'>"
    assert expected_msg == str(errorMsg.value)


def test_write_csv(tmpdir):
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                index=pd.Index(data=[1, 2, 3, 4], name="Year"))

    file = tmpdir.join('output.csv')

    dpl.writers(input_df, file.strpath[:-4], "csv")

    expected_csv_lines = ['"Year","SeriesA","SeriesB"\n',
                          '1,0.1,0.2\n',
                          '2,0.3,0.4\n',
                          '3,0.5,0.6\n',
                          '4,0.7,0.8\n']

    assert expected_csv_lines == file.readlines()



def test_write_rwl(tmpdir):
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                index=pd.Index(data=[1, 2, 3, 4], name="Year"))

    file = tmpdir.join('output.rwl')

    dpl.writers(input_df, file.strpath[:-4], "rwl")

    # default prec=0.001: values x1000, space-padded, -9999 end marker
    expected_rwl_lines = ['SeriesA    1   100   300   500   700 -9999\n',
                          'SeriesB    1   200   400   600   800 -9999\n']

    assert expected_rwl_lines == file.readlines()


def test_write_rwl_precision(tmpdir):
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                            index=pd.Index(data=[1, 2, 3, 4], name="Year"))
    # prec=0.01: values x100, 999 end marker
    file = tmpdir.join('p01.rwl')
    dpl.writers(input_df, file.strpath[:-4], "rwl", prec=0.01)
    assert file.readlines() == ['SeriesA    1    10    30    50    70   999\n',
                                'SeriesB    1    20    40    60    80   999\n']
    # invalid precision is rejected
    with pytest.raises(ValueError):
        dpl.writers(input_df, tmpdir.join('bad').strpath, "rwl", prec=0.5)


def test_write_rwl_interior_gap_sentinel_default(tmpdir):
    # Default gaps=-99 (Ed Cook's ARSTAN convention): the true interior gap gets the
    # sentinel in one continuous block, distinct from a real 0 (a locally absent
    # ring, always written as 0).
    import numpy as np
    input_df = pd.DataFrame(data={"S1": [0.10, 0.30, np.nan, 0.52]},
                            index=pd.Index(data=[1990, 1991, 1992, 1993], name="Year"))
    file = tmpdir.join('gap.rwl')
    dpl.writers(input_df, file.strpath[:-4], "rwl")           # default sentinel
    assert file.readlines() == ['S1      1990   100   300   -99   520 -9999\n']

    # a locally absent ring (real 0) is written as 0, NOT the gap sentinel
    absent = pd.DataFrame(data={"S1": [0.10, 0.0, 0.30]},
                          index=pd.Index(data=[1990, 1991, 1992], name="Year"))
    file2 = tmpdir.join('absent.rwl')
    dpl.writers(absent, file2.strpath[:-4], "rwl")
    assert file2.readlines() == ['S1      1990   100     0   300 -9999\n']

    # a different sentinel (-9) is honoured
    file3 = tmpdir.join('g9.rwl')
    dpl.writers(input_df, file3.strpath[:-4], "rwl", gaps=-9)
    assert file3.readlines() == ['S1      1990   100   300    -9   520 -9999\n']


def test_write_rwl_interior_gap_split(tmpdir):
    # gaps="split": close the block at the gap and reopen at the next present year,
    # so the gap reads back as NaN with no sentinel written in the file.
    import numpy as np
    input_df = pd.DataFrame(data={"S1": [0.10, 0.30, np.nan, 0.52]},
                            index=pd.Index(data=[1990, 1991, 1992, 1993], name="Year"))
    file = tmpdir.join('gap.rwl')
    dpl.writers(input_df, file.strpath[:-4], "rwl", gaps="split")
    assert file.readlines() == ['S1      1990   100   300 -9999\n',
                                'S1      1993   520 -9999\n']


def test_write_rwl_bad_gap_sentinel(tmpdir):
    import numpy as np
    df = pd.DataFrame(data={"S1": [0.1, np.nan, 0.3]},
                      index=pd.Index(data=[1, 2, 3], name="Year"))
    for bad in (0, 5, -9999):                       # non-negative or a stop marker
        with pytest.raises(ValueError):
            dpl.writers(df, tmpdir.join("b").strpath, "rwl", gaps=bad)
    with pytest.raises(ValueError):                 # 999 collides at prec=0.01
        dpl.writers(df, tmpdir.join("b").strpath, "rwl", prec=0.01, gaps=999)


def _crn():
    return pd.DataFrame(
        data={"std": [1.0, 0.85, 1.2, 0.9, 1.05, 1.1],
              "samp_depth": [2, 2, 3, 3, 4, 4]},
        index=pd.Index(range(1998, 2004), name="Year"))


def _hdr():
    return dict(site_id="TST01", site_name="Test Site", species_code="PSME",
                state_country="Arizona", species="Douglas-fir", elevation="2000M",
                latitude="3412", longitude="-11145", investigators="Doe J A")


def test_write_crn_header_and_values(tmpdir):
    # chronology in, correct ITRDB header + index values x1000, matching dplR's
    # encoding (std 1.0 -> 1000, missing -> 9990).
    file = tmpdir.join("out.crn")
    dpl.writers(_crn(), file.strpath[:-4], "crn", header=_hdr())
    lines = file.read().splitlines()
    assert lines[0].startswith("TST01") and lines[0].rstrip().endswith("PSME")   # rec1
    assert "1998 2003" in lines[1]                                                # years in rec2
    assert lines[2].startswith("TST01") and "Doe J A" in lines[2]                 # rec3
    data = lines[3]
    assert data.startswith("TST01")
    assert "1000  2" in data          # std 1.0 -> 1000, depth 2
    assert "9990  0" in data          # opening-decade padding uses the 9990 marker


def test_write_crn_requires_header(tmpdir):
    with pytest.raises(ValueError):
        dpl.writers(_crn(), tmpdir.join("x").strpath, "crn")             # no header
    with pytest.raises(ValueError):
        dpl.writers(_crn(), tmpdir.join("y").strpath, "crn",
                    header={"site_id": "X"})                             # incomplete


def test_write_crn_type_code(tmpdir):
    file = tmpdir.join("a.crn")
    dpl.writers(_crn(), file.strpath[:-4], "crn", header=_hdr(),
                chronology_type="arstan")
    rec2 = file.read().splitlines()[1]
    assert rec2[61:63] == "A "        # ARSTAN type code at columns 62-63


def test_samp_stats_depth_seg_age():
    import numpy as np
    df = pd.DataFrame({"S1": [0.1, 0.2, 0.3, np.nan, 0.5],
                       "S2": [np.nan, 0.5, 0.6, 0.7, 0.8],
                       "S3": [np.nan, np.nan, 0.9, 1.0, 1.1]},
                      index=pd.Index(range(1990, 1995), name="Year"))
    ss = dpl.samp_stats(df)
    assert list(ss.columns) == ["samp_depth", "seg", "age"]
    assert list(ss["samp_depth"]) == [1, 2, 3, 2, 3]        # per-year present count
    assert ss["samp_depth"].dtype.kind == "i"               # integer count
    # 1992: S1,S2,S3 present; ring counts 4,4,3 -> seg 3.667; ages 3,2,1 -> 2.0
    assert round(ss.loc[1992, "seg"], 3) == 3.667
    assert ss.loc[1992, "age"] == 2.0
    # 1993: S1 is a gap -> only S2,S3 present; ring counts 4,3 -> 3.5; ages 3,2 -> 2.5
    assert ss.loc[1993, "samp_depth"] == 2
    assert ss.loc[1993, "seg"] == 3.5
    assert ss.loc[1993, "age"] == 2.5


def test_samp_stats_rejects_non_dataframe():
    with pytest.raises(TypeError):
        dpl.samp_stats([1, 2, 3])


def test_write_txt_dataframe(tmpdir):
    # Generic tab table: author-chosen columns, year index becomes the first column.
    import numpy as np
    df = pd.DataFrame({"num": [1, 2], "std": [0.95, 1.10], "res": [np.nan, 0.4]},
                      index=pd.Index([1990, 1991], name="year"))
    file = tmpdir.join("t.txt")
    dpl.writers(df, file.strpath[:-4], "txt")
    assert file.readlines() == ["year\tnum\tstd\tres\n",
                                "1990\t1\t0.9500\tNA\n",
                                "1991\t2\t1.1000\t0.4000\n"]


def test_write_txt_list_of_series(tmpdir):
    # A list of year-aligned Series is concatenated, one column each.
    a = pd.Series([0.1, 0.2], index=[1990, 1991], name="raw")
    b = pd.Series([1.0, 1.2], index=[1990, 1991], name="std")
    file = tmpdir.join("t.txt")
    dpl.writers([a, b], file.strpath[:-4], "txt")
    assert file.readlines() == ["Year\traw\tstd\n",           # unnamed index -> "Year" (D10)
                                "1990\t0.1000\t1.0000\n",
                                "1991\t0.2000\t1.2000\n"]


def test_write_txt_options(tmpdir):
    # sep / float_format / na_rep are honoured; a non-Series list item is rejected.
    import numpy as np
    df = pd.DataFrame({"x": [0.5, np.nan]}, index=pd.Index([1, 2], name="year"))
    file = tmpdir.join("t.txt")
    dpl.writers(df, file.strpath[:-4], "txt", sep=" ", float_format="%.2f", na_rep="-9.99")
    assert file.readlines() == ["year x\n", "1 0.50\n", "2 -9.99\n"]
    with pytest.raises(TypeError):
        dpl.writers([df], tmpdir.join("b").strpath, "txt")   # list must hold Series

# ===========================================================================
# v0.9.0 audit follow-ups (A5, A7, A18, B5, B6, C1, C2, C4, C5, D9, D11, F6)
# ===========================================================================
import contextlib
import warnings
import numpy as np


def _rwl4():
    return pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                              "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                        index=pd.Index(data=[1, 2, 3, 4], name="Year"))


def test_writers_returns_path_and_prints_nothing(tmpdir, capsys):
    # D9: every writer returns the path written and no longer prints progress.
    out = dpl.writers(_rwl4(), tmpdir.join("q").strpath, "csv")
    assert out == tmpdir.join("q").strpath + ".csv"
    assert capsys.readouterr().out == ""
    out2 = dpl.writers(_rwl4(), tmpdir.join("q").strpath, "RWL")      # D11: case-insensitive
    assert out2.endswith("q.rwl")
    with pytest.raises(ValueError, match="format must be one of"):
        dpl.writers(_rwl4(), tmpdir.join("q").strpath, "ext")


def test_write_rwl_uses_read_precision_by_default(tmp_path):
    # C2: a file read at 0.01 mm is written back at 0.01 mm (same integers, same
    # stop marker), so a round trip no longer changes the archive's precision.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca = dpl.readers("tests/data/rwl/ca533.rwl")
    assert set(ca.attrs["dplpy_precision"].values()) == {0.01}
    out = dpl.writers(ca, str(tmp_path / "ca533"), "rwl")
    first = open(out).readline()
    assert first.rstrip().endswith("999") or "   999" in open(out).read()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = dpl.readers(out)
    assert set(back.attrs["dplpy_precision"].values()) == {0.01}
    pd.testing.assert_frame_equal(ca, back)
    # explicit prec still wins
    out2 = dpl.writers(ca, str(tmp_path / "ca533b"), "rwl", prec=0.001)
    assert "-9999" in open(out2).read()
    # mixed precision with no explicit prec is an error
    mixed = _rwl4()
    mixed.attrs["dplpy_precision"] = {"SeriesA": 0.01, "SeriesB": 0.001}
    with pytest.raises(ValueError, match="mixed"):
        dpl.writers(mixed, str(tmp_path / "mixed"), "rwl")
    dpl.writers(mixed, str(tmp_path / "mixed"), "rwl", prec=0.01)      # explicit: fine


def test_write_rwl_noncontiguous_index_raises(tmpdir):
    # B5: a frame with a missing year row (e.g. after dropna) is refused clearly.
    df = pd.DataFrame({"S1": [0.1, 0.2, 0.3]}, index=pd.Index([1990, 1991, 1993], name="Year"))
    with pytest.raises(ValueError, match="contiguous"):
        dpl.writers(df, tmpdir.join("gap").strpath, "rwl")


def test_write_rwl_and_csv_duplicate_columns_raise(tmpdir):
    # B6
    df = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=["S1", "S1"],
                      index=pd.Index([1990, 1991], name="Year"))
    with pytest.raises(ValueError, match="duplicate"):
        dpl.writers(df, tmpdir.join("d").strpath, "rwl")
    with pytest.raises(ValueError, match="duplicate"):
        dpl.writers(df, tmpdir.join("d").strpath, "csv")


def test_write_rwl_truncates_long_ids_uniquely(tmp_path):
    # C1: IDs longer than 8 characters are truncated with a warning (dplR), kept
    # unique, and the file reads back.
    df = pd.DataFrame({"ABCDEFGHIJ": [0.1, 0.2], "ABCDEFGHIK": [0.3, 0.4], "OK": [0.5, 0.6]},
                      index=pd.Index([1990, 1991], name="Year"))
    with pytest.warns(UserWarning, match="truncated"):
        out = dpl.writers(df, str(tmp_path / "long"), "rwl")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = dpl.readers(out)
    assert list(back.columns) == ["ABCDEFGH", "ABCDEFG2", "OK"]
    # BC years leave only 7 characters for the id
    bc = pd.DataFrame({"ABCDEFGH": [0.1, 0.2]}, index=pd.Index([-10, -9], name="Year"))
    with pytest.warns(UserWarning, match="7 characters"):
        out2 = dpl.writers(bc, str(tmp_path / "bc"), "rwl")
    assert open(out2).readline().startswith("ABCDEFG  -10")


def test_write_rwl_warns_on_negative_and_stop_code_values(tmpdir):
    # C5: negatives are written but cannot survive a Tucson round trip -> warn.
    df = pd.DataFrame({"S1": [0.1, -0.2, 0.3]}, index=pd.Index([1990, 1991, 1992], name="Year"))
    with pytest.warns(UserWarning, match="negative"):
        dpl.writers(df, tmpdir.join("neg").strpath, "rwl")
    # C4: a real 9.99 mm at prec=0.01 is the stop code -> warn, suggest 0.001.
    df2 = pd.DataFrame({"S1": [0.1, 9.99, 0.3]}, index=pd.Index([1990, 1991, 1992], name="Year"))
    with pytest.warns(UserWarning, match="prec=0.001"):
        dpl.writers(df2, tmpdir.join("stop").strpath, "rwl", prec=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dpl.writers(df2, tmpdir.join("fine").strpath, "rwl", prec=0.001)   # no warning


def test_write_rwl_decade_boundary_and_long_series_roundtrip(tmp_path):
    # F6 (array path) must reproduce the original layout exactly, including a
    # series whose last year ends a decade (stop marker on the next decade line).
    years = pd.Index(range(1875, 2000), name="Year")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"S%02d" % k: rng.uniform(0.05, 3.0, len(years)).round(3)
                       for k in range(5)}, index=years)
    df.iloc[10:13, 2] = np.nan                       # an interior gap
    df.iloc[:7, 3] = np.nan                          # a late start
    out = dpl.writers(df, str(tmp_path / "rt"), "rwl")
    lines = open(out).read().splitlines()
    assert lines[0].startswith("S00     1875")
    assert any(ln.startswith("S00     2000 -9999") for ln in lines)   # stop on the next line
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = dpl.readers(out)
    pd.testing.assert_frame_equal(back, df, check_names=False)


def _bc_crn():
    return pd.DataFrame({"std": [1.0, 0.85, 1.2, 0.9, 1.05, 1.1, 0.95, 1.0, 1.02, 0.98,
                                 1.11, 0.99, 1.0, 1.3, 0.7],
                         "samp_depth": [2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]},
                        index=pd.Index(range(-1265, -1250), name="Year"))


def test_write_crn_bc_years_roundtrip(tmp_path):
    # A5/A6: a BC chronology is written in dplR's layout (5-char id, sign in col
    # 6) and reads back with the right years and values.
    crn = _bc_crn()
    hdr = _hdr()
    with pytest.warns(UserWarning, match="truncated"):              # 'TST01' fits; 'TST01X' would not
        hdr["site_id"] = "TST01X"
        out = dpl.writers(crn, str(tmp_path / "bc"), "crn", header=hdr)
    lines = open(out).read().splitlines()
    assert lines[3].startswith("TST01-1265")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = dpl.read_crn(out)
    assert list(back.index) == list(range(-1265, -1251 + 1))
    np.testing.assert_allclose(back["std"].to_numpy(), crn["std"].round(3).to_numpy())
    np.testing.assert_array_equal(back["samp_depth"].to_numpy(), crn["samp_depth"].to_numpy())
    # a year below -9999 cannot be written
    deep = crn.copy()
    deep.index = pd.Index(range(-10005, -9990), name="Year")
    with pytest.raises(ValueError, match="5 characters"):
        dpl.writers(deep, str(tmp_path / "deep"), "crn", header=_hdr())


def test_write_crn_field_width_validation(tmpdir):
    # A7: an index >= 10 or a depth outside 0-999 raises naming the year; a NaN
    # depth is written as 0.
    crn = _crn()
    crn.loc[2000, "samp_depth"] = 1200
    with pytest.raises(ValueError, match="2000"):
        dpl.writers(crn, tmpdir.join("d").strpath, "crn", header=_hdr())
    crn = _crn()
    crn.loc[2001, "std"] = 12.5
    with pytest.raises(ValueError, match="2001"):
        dpl.writers(crn, tmpdir.join("d").strpath, "crn", header=_hdr())
    crn = _crn().astype(float)
    crn.loc[2002, "samp_depth"] = np.nan
    f = tmpdir.join("nan")
    dpl.writers(crn, f.strpath, "crn", header=_hdr())
    data = f.new(ext="crn").read().splitlines()[4]    # the 2000s decade row
    assert data.startswith("TST01 2000") and "1050  0" in data   # std 1.05 at 2002, depth 0


def test_write_crn_packs_decimal_coordinates(tmpdir):
    # A18: SiteMetadata carries decimal degrees / metres; the writer packs them
    # into the ITRDB fields (DDMM / DDDMM, '%dM') instead of writing '37.5167'.
    sm = SiteMetadata(site_id="CA533", site_name="Campito", species_code="PILO",
                      country_region="California", species_name="Pinus longaeva",
                      elevation_m=3470, latitude=37.5167, longitude=-118.2,
                      investigators="Lamarche")
    f = tmpdir.join("sm")
    dpl.writers(_crn(), f.strpath, "crn", header=sm)
    rec2 = f.new(ext="crn").read().splitlines()[1]
    assert "3470M" in rec2
    assert "3731-11812" in rec2
    assert "37.5" not in rec2
    # already-packed strings pass through unchanged; a decimal string is packed
    hdr = _hdr()
    hdr.update(latitude="1917", longitude="09856", elevation="200")
    g = tmpdir.join("str")
    dpl.writers(_crn(), g.strpath, "crn", header=hdr)
    rec2 = g.new(ext="crn").read().splitlines()[1]
    assert "1917 09856" in rec2 and " 200M" in rec2
    # a pair that cannot fit the 10-character field raises
    from dplpy.site_metadata import pack_crn_coordinates
    assert pack_crn_coordinates(-35.5, 148.2) == "-353014812"      # southern + eastern fits
    with pytest.raises(ValueError, match="10-character"):
        pack_crn_coordinates(-35.5, -148.2)


def test_write_crn_roundtrips_coordinates_through_reader(tmp_path):
    # A18 end to end: metadata read from a real header -> SiteMetadata -> .crn
    # header -> back through the Tucson header parser gives the same coordinates.
    from dplpy.readers import _extract_header_metadata
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rwl = dpl.readers("tests/data/rwl/ca667.rwl")
    sm = SiteMetadata.from_rwl(rwl)
    out = dpl.writers(_crn(), str(tmp_path / "ca667"), "crn", header=sm)
    md = _extract_header_metadata(open(out).read().splitlines()[:3])
    assert md["latitude"] == pytest.approx(sm.latitude, abs=1e-3)
    assert md["longitude"] == pytest.approx(sm.longitude, abs=1e-3)
    assert md["elevation_m"] == sm.elevation_m
