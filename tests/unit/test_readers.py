import dplpy as dpl
import pandas as pd
import numpy as np
import pytest
import io
import warnings
from unittest.mock import patch, Mock

'''
    An unrecognized suffix is no longer rejected outright -- the content is
    sniffed. A file whose format can't be inferred from suffix OR content raises
    a clear error telling the user to pass format=.
'''
def test_unknown_format_raises_helpful_error(tmp_path):
    p = tmp_path / "mystery.dat"
    p.write_text("this is not ring-width data\njust some prose here\n")
    with pytest.raises(ValueError) as e:
        dpl.readers(str(p))
    assert "format" in str(e.value).lower()


def test_bad_format_argument_raises():
    with pytest.raises(ValueError):
        dpl.readers("tests/data/rwl/ca533.rwl", format="bogus")


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


def test_reads_tucson_from_nonstandard_suffix(tmp_path):
    # A valid Tucson file with a .txt suffix is recognized by content sniffing.
    import shutil
    local = _read_quiet(RWL + "ca533.rwl")
    p = tmp_path / "ca533.txt"
    shutil.copy(RWL + "ca533.rwl", str(p))
    got = _read_quiet(str(p))
    pd.testing.assert_frame_equal(local, got)


def test_format_override_forces_tucson(tmp_path):
    # No suffix at all, but format='tucson' forces the Tucson reader.
    import shutil
    local = _read_quiet(RWL + "ca533.rwl")
    p = tmp_path / "ca533_nosuffix"
    shutil.copy(RWL + "ca533.rwl", str(p))
    got = _read_quiet(str(p), format="tucson")
    pd.testing.assert_frame_equal(local, got)


def test_reads_csv_from_nonstandard_suffix(tmp_path):
    # A CSV with a .txt suffix is sniffed as CSV, not Tucson.
    import shutil
    local = _read_quiet("tests/data/csv/ca533.csv")
    p = tmp_path / "ca533csv.txt"
    shutil.copy("tests/data/csv/ca533.csv", str(p))
    got = _read_quiet(str(p))
    pd.testing.assert_frame_equal(local, got)


def test_header_lines_skipped_reported():
    # Auto-detection records how many header lines it skipped (transparency).
    data = _read_quiet(RWL + "th001.rwl")           # 3-line header
    assert data.attrs["dplpy_header_lines_skipped"] == 3
    # a header-less file reports 0
    ca = _read_quiet(RWL + "ca533.rwl")
    assert ca.attrs["dplpy_header_lines_skipped"] == 0


# ---------------------------------------------------------------------------
# Header metadata extraction (prototype).
# ---------------------------------------------------------------------------

def test_metadata_extraction_tx042():
    md = dpl.metadata(RWL + "tx042.rwl")
    assert md["site_id"] == "BSC"
    assert md["species_code"] == "PSME"
    assert md["species_name"] == "Douglas Fir"
    assert md["country_region"] == "Texas"
    assert md["elevation_m"] == 2057
    assert md["latitude"] == pytest.approx(29.25, abs=1e-3)
    assert md["longitude"] == pytest.approx(-103.3, abs=1e-3)   # sign correct here
    assert md["first_year"] == 1473 and md["last_year"] == 1992


def test_metadata_extraction_th001():
    md = dpl.metadata(RWL + "th001.rwl")
    assert md["site_id"] == "MHGSTG"
    assert md["species_code"] == "TEGR"
    assert md["country_region"] == "Thailand"
    assert md["latitude"] == pytest.approx(19.2833, abs=1e-3)
    assert md["longitude"] == pytest.approx(98.9333, abs=1e-3)


def test_metadata_on_df_attrs():
    d = _read_quiet(RWL + "th001.rwl")
    assert "dplpy_metadata" in d.attrs
    assert d.attrs["dplpy_metadata"]["species_code"] == "TEGR"


def test_metadata_headerless_file_is_empty():
    # ca533 has no header -> all fields None, no crash.
    md = dpl.metadata(RWL + "ca533.rwl")
    assert md["n_header_lines"] == 0
    assert md["site_id"] is None and md["species_code"] is None


def test_dm_to_decimal_decoder():
    from dplpy.readers import _dm_to_decimal
    assert _dm_to_decimal("3627") == pytest.approx(36.45)
    assert _dm_to_decimal("-2053") == pytest.approx(-20.8833, abs=1e-3)
    assert _dm_to_decimal("00406") == pytest.approx(4.1, abs=1e-3)
    assert _dm_to_decimal("abc") is None


def test_metadata_hemisphere_correction_americas(tmp_path):
    # A recognized N. American state forces West longitude even when the file
    # stored it without the '-' sign.
    p = tmp_path / "arz.rwl"
    p.write_text(
        "ARZ    1 Some Arizona Site                                   PSME\n"
        "ARZ    2 Arizona      Douglas Fir       2000M  3530 11140    __    1500 1990\n"
        "ARZ    3 J. Smith\n"
        "ARZ    1500   100   200 -9999\n"
    )
    md = dpl.metadata(str(p))
    assert md["country_region"] == "Arizona"
    assert md["hemisphere_verified"] is True
    assert md["latitude"] == pytest.approx(35.5, abs=1e-2)
    assert md["longitude"] == pytest.approx(-111.6667, abs=1e-3)   # corrected to West


def test_metadata_hemisphere_unverified_eastern(tmp_path):
    # An Eastern-hemisphere country is left as decoded and flagged unverified
    # (we don't guess sign outside the Americas).
    p = tmp_path / "nor.rwl"
    p.write_text(
        "NOR    1 Some Norway Site                                    PISY\n"
        "NOR    2 Norway   Scots pine        200  6829 1602    __    1485 1978\n"
        "NOR    3 A. Person\n"
        "NOR    1485   100   200 -9999\n"
    )
    md = dpl.metadata(str(p))
    assert md["country_region"] == "Norway"
    assert md["hemisphere_verified"] is False
    assert md["longitude"] == pytest.approx(16.0333, abs=1e-3)     # left as decoded


def test_region_hemisphere_lookup():
    from dplpy.readers import _region_hemisphere
    assert _region_hemisphere("Texas") == (1, -1)
    assert _region_hemisphere("California") == (1, -1)
    assert _region_hemisphere("Chile") == (None, -1)
    assert _region_hemisphere("Norway") == (None, None)
    assert _region_hemisphere("") == (None, None)


def test_future_year_raises_in_strict_mode(tmp_path):
    # A most-recent year in the future is a sanity-check failure: a ring cannot
    # post-date the present. 2999 is safely in the future regardless of run date.
    p = tmp_path / "future.csv"
    pd.DataFrame({"Year": [2000, 2001, 2999], "S1": [0.1, 0.2, 0.3]}).to_csv(p, index=False)
    with pytest.raises(ValueError) as e:
        dpl.readers(str(p))                               # on_error='raise' (default)
    assert "future" in str(e.value) and "2999" in str(e.value)


def test_future_year_warns_and_continues_in_salvage_mode(tmp_path):
    p = tmp_path / "future.csv"
    pd.DataFrame({"Year": [2000, 2001, 2999], "S1": [0.1, 0.2, 0.3]}).to_csv(p, index=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = dpl.readers(str(p), on_error="warn")
    assert res is not None and int(res.index.max()) == 2999   # kept, not dropped
    assert any("future" in str(x.message) for x in w)


# --- fixed-width column misalignment ('1  065', shifted grids) ---------------
# A value field that still holds a space once stripped (e.g. '1  065', or az606's
# '6  142' where 126 was truncated to 12) means the fixed-width columns are shifted
# and the row is corrupt. In a fixed-width format that is severe: strict refuses
# the whole file, salvage NaNs the whole offending row and records it.
_SPLIT_RWL = [
    "S1        1 TEST SITE                      TS",
    "S1        2 TESTLAND SPECIES     1000 M ...  1900 1911",
    "S1        3 TESTER",
    "S1      1900   100   200   300   400   500   600   700   800   900   950",
    "S1      1910   1501  065   999",
]
_CLEAN_RWL = [
    "S1        1 TEST SITE                      TS",
    "S1        2 TESTLAND SPECIES     1000 M ...  1900 1902",
    "S1        3 TESTER",
    "S1      1900   100   200   999",
]


def test_embedded_space_value_raises_in_strict(tmp_path):
    # a value with embedded spaces ('1  065') means a shifted/embedded fixed-width
    # field -- a column misalignment. In a fixed-width format that is fatal, so
    # strict mode refuses the whole file.
    p = tmp_path / "split.rwl"
    p.write_text("\n".join(_SPLIT_RWL) + "\n")
    with pytest.raises(ValueError, match="misalignment"):
        dpl.readers(str(p), on_error="raise")


def test_embedded_space_value_is_nan_row_in_salvage(tmp_path):
    # salvage NaNs the whole misaligned row (once the grid shifts, the rest is
    # suspect), records it, and keeps the other decades / the rest of the batch.
    p = tmp_path / "split.rwl"
    p.write_text("\n".join(_SPLIT_RWL) + "\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = dpl.readers(str(p), on_error="warn")
    assert 1900 in df["S1"].dropna().index                    # the clean 1900 decade still read
    assert 1910 not in df["S1"].dropna().index                # the misaligned 1910 row is gone
    acts = [a for a in df.attrs["dplpy_salvage"] if a["issue"] == "column_misalignment"]
    assert acts and acts[0]["series"] == "S1"
    assert any("misaligned" in str(x.message) for x in w)


def test_clean_file_reports_no_unreadable_clause(tmp_path, capsys):
    p = tmp_path / "clean.rwl"
    p.write_text("\n".join(_CLEAN_RWL) + "\n")
    df = dpl.readers(str(p))
    out = capsys.readouterr().out
    assert df.attrs["dplpy_dropped"] == 0
    assert "unreadable" not in out                           # clause omitted when nothing was dropped


# --- fixed-width value region capped at cols 13-72; joined-record splitting ----
def _rwl_row(sid, year, vals):
    return sid.ljust(8) + ("%4d" % year) + "".join("%6d" % v for v in vals)


def test_trailing_count_column_ignored(tmp_path):
    # a nonstandard per-row value-count past col 72 (as in germ012l) must not be
    # read as an 11th value -- that used to force a whitespace fall-back and a
    # spurious self-overlap.
    lines = [
        _rwl_row("SER1", 1900, [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]) + "      10",
        _rwl_row("SER1", 1910, [200, 999]),
    ]
    p = tmp_path / "count.rwl"
    p.write_text("\n".join(lines) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = dpl.readers(str(p), header=False, on_error="raise")   # no self-overlap raised
    assert list(df.columns) == ["SER1"]
    assert df["SER1"].loc[1900] == pytest.approx(1.00)             # 100 at 0.01 mm precision
    assert int(df["SER1"].last_valid_index()) == 1910


def test_joined_records_are_split(tmp_path):
    # a missing line break joining SER1's terminal row to SER2's first row
    joined = (_rwl_row("SER1", 1900, [100, 110, 120, 130, 140, 150, 160, 170, 180, 999])
              + _rwl_row("SER2", 1910, [200, 210, 999]))
    p = tmp_path / "joined.rwl"
    p.write_text(joined + "\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = dpl.readers(str(p), header=False, on_error="raise")
    assert set(df.columns) == {"SER1", "SER2"}
    assert int(df["SER1"].last_valid_index()) == 1908             # 999 stop at 1909
    assert int(df["SER2"].first_valid_index()) == 1910
    assert any("joined record" in str(x.message) for x in w)


def test_site_id_after_stop_is_not_split(tmp_path):
    # a normal trailing site ID after the stop marker (no following year) must NOT
    # be treated as a joined record
    line = _rwl_row("SER1", 1900, [100, 110, 120, 130, 140, 150, 160, 170, 180, 999]) + "TRW1A"
    p = tmp_path / "siteid.rwl"
    p.write_text(line + "\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = dpl.readers(str(p), header=False, on_error="raise")
    assert list(df.columns) == ["SER1"]
    assert not any("joined record" in str(x.message) for x in w)


# --- guard: refuse files with more than 3 header lines (not a Tucson .rwl) ------
def test_over_long_header_is_refused(tmp_path):
    # >3 metadata lines before data -> not a clean Tucson .rwl (a NOAA Template
    # file has ~100; a doubled ITRDB header has 6). Strict raises, salvage -> None.
    for n in (4, 10):
        lines = ["metadata header line %d here" % i for i in range(1, n + 1)]
        lines += [_rwl_row("SER1", 1900, [100, 110, 999])]
        p = tmp_path / ("h%d.rwl" % n)
        p.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="does not look like a Tucson"):
            dpl.readers(str(p))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert dpl.readers(str(p), on_error="warn") is None  # salvage -> None, not garbage


def test_three_line_header_reads(tmp_path):
    # the standard 3-line ITRDB header is the maximum and must still read
    lines = ["metadata header line %d here" % i for i in range(1, 4)]     # 3 header lines
    lines += [_rwl_row("SER1", 1900, [100, 110, 120, 999])]
    p = tmp_path / "std_header.rwl"
    p.write_text("\n".join(lines) + "\n")
    df = dpl.readers(str(p))
    assert list(df.columns) == ["SER1"]
    assert df.attrs["dplpy_header_lines_skipped"] == 3


def test_header_false_bypasses_the_guard(tmp_path):
    # header=False (caller is sure) must not trip the long-header guard
    lines = ["metadata header line %d here" % i for i in range(1, 11)]
    lines += [_rwl_row("SER1", 1900, [100, 110, 999])]
    p = tmp_path / "forced.rwl"
    p.write_text("\n".join(lines) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dpl.readers(str(p), header=False, on_error="warn")
        except ValueError as e:
            assert "does not look like a Tucson" not in str(e)     # guard not applied


# --- '#' inside a series ID is data, not a comment (real ITRDB: SP#1, GFI..#H) --
def test_hash_in_series_id_is_not_a_comment(tmp_path):
    lines = [
        "# a genuine comment line at the top",           # starts with '#' -> stripped
        _rwl_row("SP#1", 1900, [100, 110, 999]),         # '#' inside the ID -> data, kept
        _rwl_row("SP#2", 1900, [120, 130, 999]),
    ]
    p = tmp_path / "hashid.rwl"
    p.write_text("\n".join(lines) + "\n")
    df = dpl.readers(str(p), on_error="warn")
    assert "SP#1" in df.columns and "SP#2" in df.columns   # not dropped as comments
    assert df.loc[1900, "SP#1"] == pytest.approx(1.00)
    assert df.loc[1901, "SP#2"] == pytest.approx(1.30)


def test_hash_marked_note_row_is_stripped_not_counted_as_header(tmp_path):
    # a '#### ...' annotation row (CDendro-style, as in ita065/prt004) sits between
    # the header and data; it doesn't start with '#' and doesn't parse as data, so
    # it must be stripped as a comment -- not counted as a 4th header line (which
    # would trip the >3 guard).
    lines = ["metadata header line %d here" % i for i in range(1, 4)]    # 3 real headers
    lines += ["SER1    #### corrC GT 0.7, CDendro note;"]                # annotation row
    lines += [_rwl_row("SER1", 1900, [100, 110, 999])]
    p = tmp_path / "noterow.rwl"
    p.write_text("\n".join(lines) + "\n")
    df = dpl.readers(str(p))                                             # not guard-rejected
    assert list(df.columns) == ["SER1"]
    assert df.attrs["dplpy_header_lines_skipped"] == 3                   # note row not a header


def test_999_inside_series_id_is_not_split(tmp_path):
    # a series ID containing '999' (e.g. 'S999A', as in aus118/aus123's Huon pine
    # cores) must NOT be split by the joined-record repair -- the '999' is in the
    # ID field (column < 12), not a stop marker in the value region.
    lines = ["metadata header line %d here" % i for i in range(1, 4)]
    lines += [_rwl_row("S999A", 1900, [100, 110, 120, 999])]
    p = tmp_path / "s999.rwl"
    p.write_text("\n".join(lines) + "\n")
    df = dpl.readers(str(p))
    assert "S999A" in df.columns and "S999" not in df.columns   # kept whole, not split
    assert df.loc[1900, "S999A"] == pytest.approx(1.00)
