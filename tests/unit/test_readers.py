import dplpy as dpl
import pandas as pd
import numpy as np
import pytest
import io
import warnings
from unittest.mock import patch, Mock

'''
    Test that when given an incorrect file extension, program raises 
    an error with expected message.
'''
def test_wrong_file_extension():
    with pytest.raises(ValueError) as errorMsg:
        dpl.readers("filename.txt")

    wrong_ext_msg = """

Unable to read file, please check that you're using a supported type
Accepted file types are .csv and .rwl

Example usages:
>>> import dplpy as dpl
>>> data = dpl.readers('../tests/data/csv/filename.csv')
>>> data = dpl.readers('../tests/data/rwl/filename.rwl'), header=True
"""
    assert wrong_ext_msg == str(errorMsg.value)


'''
    Mocks output of pd.read_csv to return appropriate dataframe only if the
    parameter used is the expected file name.
'''
def mock_read_csv_output(file_path, skiprows=0):
    if file_path == "correct_file.csv":
        return pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8], 
                                  "Year": [1, 2, 3, 4]})
    return None

'''
    Mocks output of builtins.open to return an io.TextIOWrapper object that contains the lines
    that will be read for processing
'''
def mock_open_output(file_path, open_type):
    # Verify that file is opened in read mode and read mode only
    if open_type != "r":
        wrapper = io.TextIOWrapper(
            io.UnsupportedOperation(),
            encoding='cp1252',
            line_buffering=True,
        )

        wrapper.mode = open_type
        return wrapper
    
    output  = io.BytesIO()
    wrapper = io.TextIOWrapper(
        output,
        encoding='cp1252',
        line_buffering=True,
    )

    if file_path == "valid_rwl_correct_format.rwl":
        wrapper.write("SeriesA 1       10    30    50    70   999\n")
        wrapper.write("SeriesB 1      200   400   600   800 -9999\n")
        wrapper.seek(0,0)
    elif file_path == "valid_rwl_with_headers.rwl":
        wrapper.write("Header line 1\n")
        wrapper.write("Header line 2\n")
        wrapper.write("Header line 3\n")
        wrapper.write("SeriesA 1       10    30    50    70   999\n")
        wrapper.write("SeriesB 1      200   400   600   800 -9999\n")
        wrapper.seek(0,0)
    elif file_path == "valid_rwl_with_blanks.rwl":
        wrapper.write("SeriesA 1       10    30    50    70   999\n")
        wrapper.write("                                          \n")
        wrapper.write("SeriesB 1      200   400   600   800 -9999\n")
        wrapper.seek(0,0)
    else:
        raise OSError("File not found")

    wrapper.mode = open_type

    return wrapper 

'''
    Given input file.csv, test that readers produces the expected dataframe.
'''
@patch('pandas.read_csv')
def test_correct_csv_format(mock_read_csv: Mock):
    mock_read_csv.side_effect = mock_read_csv_output
    results = dpl.readers("correct_file.csv")
    mock_read_csv.assert_called_once_with("correct_file.csv", skiprows=0)

    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]}, 
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year")
                                    )
    pd.testing.assert_frame_equal(results, expected_df)

'''
    Given input file valid_rwl_correct_format.rwl, test that readers produces
    the expected dataframe.
'''
@patch('builtins.open')
def test_correct_rwl_format(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    results = dpl.readers("valid_rwl_correct_format.rwl")
    mock_open.assert_called_once_with("valid_rwl_correct_format.rwl", "r")

    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))
    print(results)
    pd.testing.assert_frame_equal(results, expected_df)

'''
    Given input valid_rwl_correct_format.rwl, and skip_lines=1, test that readers
    produces the expected dataframe.
'''
@patch('builtins.open')
def test_correct_rwl_skip_lines(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    results = dpl.readers("valid_rwl_correct_format.rwl", skip_lines=1)
    print(results)
    mock_open.assert_called_once_with("valid_rwl_correct_format.rwl", "r")

    expected_df = pd.DataFrame(data={"SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))
    pd.testing.assert_frame_equal(results, expected_df)

'''
    Given input valid_rwl_correct_format.rwl, and header=True, test that readers
    correctly skips header lines to produce the expected dataframe.
'''
@patch('builtins.open')
def test_correct_rwl_with_headers(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    results = dpl.readers("valid_rwl_with_headers.rwl", header=True)
    mock_open.assert_called_once_with("valid_rwl_with_headers.rwl", "r")

    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))
    pd.testing.assert_frame_equal(results, expected_df)

@patch('builtins.open')
def test_rwl_with_blank_lines(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    expected_warning = "Empty line found at line 2"
    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))

    with pytest.warns(UserWarning, match=expected_warning):
        results = dpl.readers("valid_rwl_with_blanks.rwl")
        mock_open.assert_called_once_with("valid_rwl_with_blanks.rwl", "r")
        pd.testing.assert_frame_equal(results, expected_df)


# ===========================================================================
# Hardened Tucson (.rwl) reader (v0.2.x): header auto-detection, dplR fidelity,
# deliberate NaN divergences (gaps / anomalous negatives), and duplicate-ID
# rejection. Reference values are hardcoded from dplR 1.7.9 read.rwl() so the
# validation runs in CI without requiring R.
# ===========================================================================

RWL = "tests/data/rwl/"


def _read_quiet(path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dpl.readers(path, **kw)


def test_rwl_autodetect_matches_dplR_ca533():
    # Read with NO header argument (auto-detection) and check exact dplR values.
    data = _read_quiet(RWL + "ca533.rwl")
    assert data.shape == (1358, 34)
    # dplR read.rwl() reference cells for series CAM011:
    assert data.loc[1530, "CAM011"] == pytest.approx(1.04)
    assert data.loc[1800, "CAM011"] == pytest.approx(0.44)
    assert data.loc[1900, "CAM011"] == pytest.approx(0.45)
    assert data.loc[1983, "CAM011"] == pytest.approx(0.68)
    # outside the series span -> NaN (dplR agrees)
    assert np.isnan(data.loc[1500, "CAM011"])


def test_rwl_header_autodetected_by_default():
    # th001 has a 3-line header; before hardening this failed unless header=True
    # was passed. Auto-detection must now read it with no header argument.
    data = _read_quiet(RWL + "th001.rwl")
    assert data.shape == (448, 77)
    assert "PATUNG" in data.columns


def test_rwl_duplicate_id_raises_and_names_series():
    # viet001 contains the duplicated ID BDF02A (two cores sharing a code).
    # dplPy must refuse rather than silently merge, and name the offender.
    with pytest.raises(ValueError) as e:
        _read_quiet(RWL + "viet001.rwl")
    msg = str(e.value)
    assert "Duplicate series ID" in msg
    assert "BDF02A" in msg


def test_rwl_anomalous_negative_becomes_nan_with_warning():
    # th001's PATUNG decade-1920 row carries a spurious -2599; it must become
    # NaN (not -2.599, not 0.0) and the user must be warned.
    with pytest.warns(UserWarning, match="anomalous negative"):
        data = dpl.readers(RWL + "th001.rwl")
    assert np.isnan(data.loc[1928, "PATUNG"])       # the anomalous value -> NaN
    assert data.loc[1927, "PATUNG"] == pytest.approx(0.983)  # neighbours intact
    assert data.loc[1929, "PATUNG"] == pytest.approx(0.597)


def test_rwl_internal_gap_stays_nan_not_zero():
    # ca667's ST850A has a real ~250-year data gap. dplR fills it with 0.0
    # (a zero-init artifact); dplPy deliberately keeps genuine gaps as NaN so
    # they stay out of downstream means. Valid values still match dplR exactly.
    data = _read_quiet(RWL + "ca667.rwl")
    assert data.loc[-390, "ST850A"] == pytest.approx(0.35)   # matches dplR
    assert data.loc[-389, "ST850A"] == pytest.approx(0.36)   # matches dplR
    assert np.isnan(data.loc[-388, "ST850A"])                # stop-marker slot
    assert np.isnan(data.loc[-385, "ST850A"])                # inside the gap


def test_rwl_self_overlap_reported_not_as_duplicate(tmp_path):
    # akfirmc-style hidden error: a single series whose opening partial-decade
    # row carries one value too many (6 values from a 1205 start), so it runs
    # into the next row that begins at 1210. This must be reported as the series
    # overlapping ITSELF (pointing at the offending row), NOT as a duplicate ID
    # -- there is only one block for this series.
    p = tmp_path / "selfoverlap.rwl"
    p.write_text(
        "AAA01   1205   206   144   216   316   308   420\n"
        "AAA01   1210   732   500   642   784   816   470   446   474   432   519\n"
        "AAA01   1220   140   160 -9999\n"
    )
    with pytest.raises(ValueError) as e:
        dpl.readers(str(p))
    msg = str(e.value)
    assert "overlaps itself" in msg          # correctly diagnosed as self-overlap
    assert "AAA01" in msg
    assert "1210" in msg                     # names the overlapping year
    assert "Duplicate series ID" not in msg  # NOT mislabelled as a duplicate


def test_rwl_999_is_real_value_in_thousandths_series(tmp_path):
    # In a 0.001 mm series (terminated by -9999), the token 999 is a real
    # 0.999 mm ring, NOT a stop marker. Precision is taken from the series
    # terminator, so a mid-series 999 must be kept as 0.999 (the older
    # "any 999 is a marker" logic silently dropped it to NaN). dplR keeps it.
    p = tmp_path / "thousandths.rwl"
    p.write_text(
        "AAA01   1900   500   600   999   700   800   650   720   540   610   590\n"
        "AAA01   1910   480   999   520   540   560   580   600   620   640   660\n"
        "AAA01   1920   700 -9999\n"
    )
    d = dpl.readers(str(p))
    assert d.loc[1902, "AAA01"] == pytest.approx(0.999)   # kept, not NaN
    assert d.loc[1911, "AAA01"] == pytest.approx(0.999)
    assert d.loc[1900, "AAA01"] == pytest.approx(0.5)
    assert d.loc[1920, "AAA01"] == pytest.approx(0.7)


def test_rwl_measurement_precision_shift_raises(tmp_path):
    # A single series measured at two precisions -- an early 0.001 mm segment
    # (ending in -9999) then a later 0.01 mm segment (ending in 999), the kok3a
    # pattern from kyrg014. Reading it at one precision makes part of it 10x
    # wrong, so normal mode must refuse and name the series.
    p = tmp_path / "shift.rwl"
    p.write_text(
        "AAA01   1900   910  1180  1480  1270 -9999\n"
        "AAA01   1910    24    20    15    23   999\n"
    )
    with pytest.raises(ValueError) as e:
        dpl.readers(str(p))
    msg = str(e.value)
    assert "precision" in msg.lower()
    assert "AAA01" in msg


def test_rwl_recovers_first_data_row_with_hash_in_header():
    # nm580's header line 1 contains 'UAFACC#=04-223'. The '#' made the older
    # reader treat that header line as a comment, after which "skip exactly 3"
    # ate the first real data row (BCS05B's 1370 decade). Skipping to the first
    # *data* line instead recovers it. dplR 1.7.9 still drops this row; dplPy now
    # reads it (matching OpenDendro's corrected read.tucson2). Raw row values ÷1000.
    d = _read_quiet(RWL + "nm580.rwl")
    assert d.loc[1370, "BCS05B"] == pytest.approx(1.373)
    assert d.loc[1379, "BCS05B"] == pytest.approx(0.784)


def test_rwl_single_line_header_keeps_all_rows(tmp_path):
    # A 1-line header must not cost the first two data rows (the old skip-3 bug).
    p = tmp_path / "h1.rwl"
    p.write_text(
        "ONE LINE OF HEADER TEXT SITE SPECIES INVESTIGATOR\n"
        "AAA01   1900   100   200   300   400   500   600   700   800   900  1000\n"
        "AAA01   1910   110 -9999\n"
    )
    d = dpl.readers(str(p))
    assert 1900 in d.index and 1909 in d.index
    assert d.loc[1900, "AAA01"] == pytest.approx(0.1)


def test_rwl_blank_interior_field_preserves_alignment(tmp_path):
    # A blank 6-char field (a missing ring written as blanks) must leave the rest
    # of the decade aligned, not shift every later value one year to the left.
    p = tmp_path / "blank.rwl"
    p.write_text(
        "AAA01   1900   100   200         400   500   600   700   800   900  1000\n"
        "AAA01   1910  1100 -9999\n"
    )
    d = dpl.readers(str(p))
    assert d.loc[1901, "AAA01"] == pytest.approx(0.2)
    assert np.isnan(d.loc[1902, "AAA01"])          # the blank -> missing
    assert d.loc[1903, "AAA01"] == pytest.approx(0.4)   # NOT shifted into 1902
    assert d.loc[1909, "AAA01"] == pytest.approx(1.0)   # last value not lost


def test_rwl_trailing_notes_are_ignored(tmp_path):
    # Notes appended after the data columns must not break the read.
    p = tmp_path / "notes.rwl"
    p.write_text(
        "AAA01   1900   100   200   300   400   500 -9999   note: suspect core\n"
        "AAA01B  1900   111   222 -9999\n"
    )
    d = dpl.readers(str(p))
    assert d.loc[1900, "AAA01"] == pytest.approx(0.1)
    assert d.loc[1904, "AAA01"] == pytest.approx(0.5)


def test_rwl_identical_duplicate_rows_deduped(tmp_path):
    # A byte-identical duplicated row (copy-paste) is unambiguous: dedupe + warn,
    # do NOT fail.
    p = tmp_path / "dup.rwl"
    p.write_text(
        "AAA01   1900   100   200   300 -9999\n"
        "AAA01   1900   100   200   300 -9999\n"
    )
    with pytest.warns(UserWarning, match="identical duplicate"):
        d = dpl.readers(str(p))
    assert d.loc[1900, "AAA01"] == pytest.approx(0.1)


def test_rwl_conflicting_duplicate_still_raises(tmp_path):
    # Same core+year with a DIFFERENT value is a genuine conflict -> still fail.
    p = tmp_path / "conf.rwl"
    p.write_text(
        "AAA01   1900   100   200   300 -9999\n"
        "AAA01   1900   555   200   300 -9999\n"
    )
    with pytest.raises(ValueError):
        dpl.readers(str(p))


def test_rwl_bunched_negative_year_uses_long_format(tmp_path):
    # A BC year < -999 written bunched to the ID with no space, e.g.
    # 'MNP262M-1270' = year -1270 (the chin067 / long-negative-years case).
    # Must be read as a 7-char ID + 5-char year, not as ID 'MNP262M-' + year
    # +1270 (which would misorder the series and trip the self-overlap check).
    p = tmp_path / "bc.rwl"
    p.write_text(
        "MNP262M-1270   375   250   300   280   310   295   330   340   360   355\n"
        "MNP262M-1260   245   240   290   220   320   280   215   300   410   500\n"
        "MNP262M-1250   170 -9999\n"
    )
    d = dpl.readers(str(p))
    assert "MNP262M" in d.columns              # 7-char ID, not 'MNP262M-'
    assert d.index.min() == -1270              # negative year parsed correctly
    assert d.loc[-1270, "MNP262M"] == pytest.approx(0.375)
    assert d.loc[-1261, "MNP262M"] == pytest.approx(0.355)


# ---------------------------------------------------------------------------
# Salvage mode (on_error="warn"): recover as much as possible instead of raising.
# ---------------------------------------------------------------------------

def test_salvage_self_overlap_drops_series_keeps_rest(tmp_path):
    p = tmp_path / "so.rwl"
    p.write_text(
        "AAA01   1205   206   144   216   316   308   420\n"   # 6 vals -> overruns to 1210
        "AAA01   1210   732   500   642 -9999\n"
        "BBB01   1900   100   200   300 -9999\n"               # clean series
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dpl.readers(str(p), on_error="warn")
    assert "AAA01" not in d.columns        # bad series dropped
    assert "BBB01" in d.columns            # clean series kept
    rep = d.attrs["dplpy_salvage"]
    assert any(r["series"] == "AAA01" and r["issue"] == "self_overlap"
               and r["action"] == "dropped" for r in rep)


def test_salvage_precision_shift_drops_series_keeps_rest(tmp_path):
    p = tmp_path / "ps.rwl"
    p.write_text(
        "AAA01   1900   910  1180 -9999\n"   # 0.001 segment...
        "AAA01   1910    24    20   999\n"    # ...then 0.01 -> precision shift
        "BBB01   1900   100   200   300 -9999\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dpl.readers(str(p), on_error="warn")
    assert "AAA01" not in d.columns
    assert "BBB01" in d.columns
    assert any(r["issue"] == "precision_shift" for r in d.attrs["dplpy_salvage"])


def test_salvage_duplicate_id_renames_and_keeps_both():
    # viet001 has BDF02A used by two overlapping cores. Salvage keeps both,
    # renaming the second block; strict still refuses.
    with pytest.raises(ValueError):
        dpl.readers(RWL + "viet001.rwl")                       # strict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dpl.readers(RWL + "viet001.rwl", on_error="warn")  # salvage
    assert "BDF02A" in d.columns and "BDF02A2" in d.columns
    assert any(r["issue"] == "duplicate_id" and r["action"].startswith("renamed")
               for r in d.attrs["dplpy_salvage"])


def test_salvage_does_not_split_disjoint_segments(tmp_path):
    # One ID entered as two disjoint segments (a gap) must MERGE, not be renamed.
    p = tmp_path / "seg.rwl"
    p.write_text(
        "AAA01   1900   100   200   300 -9999\n"
        "AAA01   2000   400   500   600 -9999\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dpl.readers(str(p), on_error="warn")
    assert "AAA01" in d.columns and "AAA012" not in d.columns   # merged, not split
    assert d.loc[1900, "AAA01"] == pytest.approx(0.1)
    assert d.loc[2000, "AAA01"] == pytest.approx(0.4)
    assert d.attrs["dplpy_salvage"] == []


def test_salvage_identical_overlap_not_renamed(tmp_path):
    # Same ID in two blocks covering the same years with IDENTICAL values (a
    # copy-paste, the cana326 case) must dedup, not rename/split.
    p = tmp_path / "iddup.rwl"
    p.write_text(
        "AAA01   1900   100   200   300 -9999\n"
        "BBB01   1910   111   222 -9999\n"
        "AAA01   1900   100   200   300 -9999\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dpl.readers(str(p), on_error="warn")
    assert "AAA01" in d.columns and "AAA012" not in d.columns
    assert not any(r["issue"] == "duplicate_id" for r in d.attrs["dplpy_salvage"])


def test_salvage_invalid_on_error_value():
    with pytest.raises(ValueError):
        dpl.readers(RWL + "ca533.rwl", on_error="bogus")


def test_readers_reads_rwl_from_url():
    # readers() accepts an http(s) URL directly and routes it through the same
    # pipeline as a local file. urllib is mocked to serve a local file's bytes
    # (no network), and the result must equal the local read.
    import urllib.request
    from unittest.mock import patch, MagicMock
    local = _read_quiet(RWL + "ca533.rwl")
    raw = open(RWL + "ca533.rwl", "rb").read()

    def fake_urlopen(url, *a, **k):
        m = MagicMock()
        m.read.return_value = raw
        m.__enter__.return_value = m
        m.__exit__.return_value = False
        return m

    # Patch urllib.request.urlopen directly (the object readers.py resolves at
    # call time). A string target like "dplpy.readers.urllib..." is fragile:
    # the package re-exports the `readers` function, so the name `dplpy.readers`
    # can resolve to the function rather than the module (version-dependent).
    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        via_url = _read_quiet(
            "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/"
            "northamerica/usa/ca533.rwl"
        )
    pd.testing.assert_frame_equal(local, via_url)
