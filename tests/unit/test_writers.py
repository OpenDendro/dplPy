import dplpy as dpl
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
    expected_msg = "Expected input data to be pandas dataframe, not <class 'pandas.core.series.Series'>"
    assert expected_msg == str(errorMsg.value)


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
    assert file.readlines() == ["year\traw\tstd\n",           # unnamed index -> "year"
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