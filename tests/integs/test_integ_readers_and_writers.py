import dplpy as dpl
import pandas as pd
import os
import warnings

def test_read_and_write_csv(tmp_path):
    ca533 = dpl.readers("./tests/data/csv/ca533.csv")

    write_path = os.path.join(tmp_path,"test_write")
    
    dpl.writers(ca533, write_path, "csv")

    ca533_alt = dpl.readers(write_path + ".csv")

    pd.testing.assert_frame_equal(ca533, ca533_alt)


def test_read_and_write_rwl_no_headers(tmp_path):
    # cana209 is a clean, header-less Tucson file, exercising the no-header
    # read/write round-trip. (viet001 was used here previously but is genuinely
    # malformed -- it contains the duplicated series ID BDF02A -- so the hardened
    # reader now correctly refuses it; that duplicate-ID behaviour is covered by
    # the unit test test_rwl_duplicate_id_raises_and_names_series.)
    cana209 = dpl.readers("./tests/data/rwl/cana209.rwl")

    write_path = os.path.join(tmp_path, "test_write")

    dpl.writers(cana209, write_path, "rwl")

    cana209_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(cana209, cana209_alt)


def test_read_and_write_rwl_with_headers(tmp_path):
    # th001 has an anomalous negative (PATUNG@1928 -> NaN + warning), asserted in
    # the unit tests; silence it here so the round-trip test output stays clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th001 = dpl.readers("./tests/data/rwl/th001.rwl", header=True)

        write_path = os.path.join(tmp_path, "test_write")

        dpl.writers(th001, write_path, "rwl")

        th001_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(th001, th001_alt)


def test_read_and_write_rwl_gap_sentinel(tmp_path):
    # Writing interior gaps as Ed Cook's negative sentinel (gaps=-99) and reading
    # back restores them as NaN, while a real 0 (locally absent ring) is preserved.
    # The reader flags the sentinels as anomalous negatives, hence the warning
    # filter here.
    import numpy as np
    df = pd.DataFrame({"S1": [0.10, 0.30, np.nan, np.nan, 0.52],
                       "S2": [0.20, 0.00, 0.40, 0.55, 0.70]},
                      index=pd.Index(range(1990, 1995), name="Year")).astype(float)
    write_path = os.path.join(tmp_path, "test_write")
    dpl.writers(df, write_path, "rwl", gaps=-99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = dpl.readers(write_path + ".rwl")
    pd.testing.assert_frame_equal(back.reindex(index=df.index, columns=df.columns), df)


def test_read_and_write_long_rwl(tmp_path):
    # ca667 has a real interior gap; the default writer marks it with the -99
    # sentinel, which the reader restores to NaN (and flags as an anomalous
    # negative -- silence that expected warning for a clean round-trip check).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True)

        write_path = os.path.join(tmp_path, "test_write")

        dpl.writers(ca667, write_path, "rwl")

        ca667_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(ca667, ca667_alt)

def test_read_and_write_weird_rwl(tmp_path):
    wwr = dpl.readers("./tests/data/rwl/wwr.rwl")

    write_path = os.path.join(tmp_path, "test_write")

    dpl.writers(wwr, write_path, "rwl")

    wwr_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(wwr, wwr_alt)

def test_read_and_write_rwl_with_blanks(tmp_path):
    # nm580l contains a blank line (-> "Empty line found" warning); silence it
    # here -- the blank-line warning is asserted in the unit tests.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nm580 = dpl.readers("./tests/data/rwl/nm580l.rwl", header=True)

        write_path = os.path.join(tmp_path, "test_write")

        dpl.writers(nm580, write_path, "rwl")

        nm580_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(nm580, nm580_alt)

# --------------------------------------------------------------------------- #
# Combined duplicate-ID series reporting: a series ID split into non-overlapping
# blocks is merged into one series, and this is now reported on a successful read
# (df.attrs['dplpy_combined']) rather than happening silently.
# --------------------------------------------------------------------------- #
import io
import contextlib


def test_combined_duplicate_series_reported_on_read():
    # ca667 contains ST850A split into two disjoint segments that merge into one.
    with contextlib.redirect_stdout(io.StringIO()):
        df = dpl.readers("./tests/data/rwl/ca667.rwl")
    combined = df.attrs.get("dplpy_combined", [])
    assert any(c["series"] == "ST850A" and c["n_blocks"] == 2 for c in combined)
    rec = [c for c in combined if c["series"] == "ST850A"][0]
    # the reported span matches the merged column's actual extent
    col = df["ST850A"].dropna()
    assert rec["first_year"] == int(col.index.min())
    assert rec["last_year"] == int(col.index.max())


def test_normal_file_reports_no_combined_series():
    with contextlib.redirect_stdout(io.StringIO()):
        df = dpl.readers("./tests/data/rwl/co021.rwl")
    assert df.attrs.get("dplpy_combined", []) == []


def test_combined_series_printed_in_summary():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dpl.readers("./tests/data/rwl/ca667.rwl")
    out = buf.getvalue()
    assert "combined from non-overlapping duplicate IDs" in out
    assert "ST850A" in out and "segments spanning" in out
