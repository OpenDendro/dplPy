import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


# --- helpers to build synthetic Tucson .crn content --------------------------
def _row(site, decade, pairs, label=None):
    """One decadal record: site(6) + decade(4) + 10x (I4 value, I3 depth)."""
    cells = []
    for v, d in pairs:
        cells.append("9990  0" if v is None else ("%4d%3d" % (v, d)))
    while len(cells) < 10:
        cells.append("9990  0")
    s = site.ljust(6) + ("%4d" % decade) + "".join(cells[:10])
    if label:
        s += "  " + label
    return s


_HDR = ["SITEAB 1 A TEST SITE NAME", "SITEAB 2 COUNTRY  SPP", "SITEAB 3 AUTHOR"]


def _read(tmp_path, name, lines, **kw):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return dpl.read_crn(str(p), **kw)


# --- standard single chronology ----------------------------------------------
def test_standard_single_crn_scaling_and_nan(tmp_path):
    lines = _HDR + [
        _row("SITEAB", 1900, [(1000, 5), (1100, 6), (950, 6)]),  # 1900,1901,1902 then 9990
        _row("SITEAB", 1910, [(1005, 7), (990, 7)]),
    ]
    df = _read(tmp_path, "site.crn", lines)
    assert list(df.columns) == ["std", "samp_depth"]
    assert df.loc[1900, "std"] == pytest.approx(1.000)      # 1000 / 1000
    assert df.loc[1901, "std"] == pytest.approx(1.100)
    assert df.loc[1900, "samp_depth"] == 5
    assert np.isnan(df.loc[1903, "std"])                    # 9990 -> NaN, kept (contiguous)
    assert int(df.index.min()) == 1900 and int(df.index.max()) == 1911
    assert list(df.index) == list(range(1900, 1912))        # contiguous index


def test_partial_first_decade_floors_correctly(tmp_path):
    # decade field 1902 floors to 1900; the two leading pad cells are missing
    lines = _HDR + [_row("SITEAB", 1902, [(None, 0), (None, 0), (1234, 3), (1250, 3)])]
    df = _read(tmp_path, "site.crn", lines)
    assert not np.isnan(df.loc[1902, "std"])                # 1902 is the 3rd cell
    assert df.loc[1902, "std"] == pytest.approx(1.234)
    assert int(df.index.min()) == 1902                      # leading all-NaN years trimmed


def test_headerless_file_reads(tmp_path):
    lines = [_row("SITEAB", 1900, [(1000, 4), (1010, 4)])]  # no 3-line header
    df = _read(tmp_path, "nohdr.crn", lines)
    assert list(df.columns) == ["std", "samp_depth"]
    assert df.loc[1900, "std"] == pytest.approx(1.0)


# --- embedded statistics line ------------------------------------------------
def test_stats_line_is_skipped_and_captured(tmp_path):
    lines = _HDR + [
        _row("SITEAB", 1900, [(1000, 5), (1100, 5)]),
        "SITEAB 783  .106  .358  .369 1.000",               # stats line (has decimals)
    ]
    df = _read(tmp_path, "site.crn", lines)
    assert int(df.index.max()) == 1901                      # stats line NOT a data row
    assert df.attrs["dplpy_crn_stats"] and ".106" in df.attrs["dplpy_crn_stats"][0]


# --- combined: multiple types, one site (ARSTAN std/res/ars) ------------------
def test_multi_type_one_site_shares_depth(tmp_path):
    depth = [(1000, 5), (1100, 6)]
    lines = (_HDR + [_row("SITEAB", 1900, depth, label="std")]
             + _HDR + [_row("SITEAB", 1900, [(980, 5), (1020, 6)], label="res")]
             + _HDR + [_row("SITEAB", 1900, [(990, 5), (1010, 6)], label="ars")])
    df = _read(tmp_path, "arstan.crn", lines)
    # typed value columns + a SINGLE shared samp_depth (depths identical)
    assert list(df.columns) == ["std", "res", "ars", "samp_depth"]
    assert df.loc[1900, "std"] == pytest.approx(1.0)
    assert df.loc[1900, "res"] == pytest.approx(0.98)
    assert df.loc[1901, "samp_depth"] == 6
    assert df.attrs["dplpy_crn"]["n_chronologies"] == 3


def test_multi_type_differing_depth_keeps_per_type_depth(tmp_path):
    lines = (_HDR + [_row("SITEAB", 1900, [(1000, 5)], label="std")]
             + _HDR + [_row("SITEAB", 1900, [(980, 9)], label="res")])  # different depth
    df = _read(tmp_path, "arstan.crn", lines)
    assert "std samp_depth" in df.columns and "res samp_depth" in df.columns
    assert df.loc[1900, "std samp_depth"] == 5
    assert df.loc[1900, "res samp_depth"] == 9


# --- combined: multiple sites ------------------------------------------------
def _multi_site_lines():
    return (["SITE_A 1 x", "SITE_A 2 x", "SITE_A 3 x",
             _row("SITE_A", 1900, [(1000, 4), (1010, 4)], label="ssfcrn")]
            + ["SITE_B 1 x", "SITE_B 2 x", "SITE_B 3 x",
               _row("SITE_B", 1950, [(970, 8), (990, 8)], label="ssfcrn")])


def test_multi_site_columns_per_site(tmp_path):
    df = _read(tmp_path, "all.crn", _multi_site_lines())
    assert list(df.columns) == ["SITE_A", "SITE_A samp_depth",
                                "SITE_B", "SITE_B samp_depth"]
    assert df.loc[1900, "SITE_A"] == pytest.approx(1.0)
    assert df.loc[1950, "SITE_B"] == pytest.approx(0.97)
    assert np.isnan(df.loc[1900, "SITE_B"])                 # different spans, union index
    assert int(df.index.min()) == 1900 and int(df.index.max()) == 1951


def test_split_by_site_returns_dict_of_frames(tmp_path):
    result = _read(tmp_path, "all.crn", _multi_site_lines(), split_by_site=True)
    assert isinstance(result, dict)
    assert list(result.keys()) == ["SITE_A", "SITE_B"]
    # each site is framed as a single-site file: typed value col + shared depth
    assert list(result["SITE_A"].columns) == ["ssfcrn", "samp_depth"]
    assert result["SITE_A"].loc[1900, "ssfcrn"] == pytest.approx(1.0)
    assert list(result["SITE_A"].index) == [1900, 1901]     # trimmed to its own span
    assert result["SITE_B"].loc[1950, "ssfcrn"] == pytest.approx(0.97)
    assert result["SITE_B"].attrs["dplpy_crn"]["sites"] == ["SITE_B"]


def test_split_by_site_single_site_is_one_entry(tmp_path):
    lines = _HDR + [_row("SITEAB", 1900, [(1000, 5)], label="std")]
    result = _read(tmp_path, "one.crn", lines, split_by_site=True)
    assert list(result.keys()) == ["SITEAB"]
    assert list(result["SITEAB"].columns) == ["std", "samp_depth"]


# --- round-trip with dpl.writers(..., "crn") ---------------------------------
def test_roundtrip_with_writer(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = dpl.readers("./tests/data/csv/ca533.csv")
        crn = dpl.chron(data, biweight=True, prewhiten=False, plot=False)
    out = str(tmp_path / "ca533")
    header = {"site_id": "CA533", "site_name": "CAMPITO MOUNTAIN", "species_code": "PILO",
              "state_country": "CALIFORNIA", "species": "BRISTLECONE", "elevation": "3400M",
              "latitude": "3730", "longitude": "-11813", "investigators": "LAMARCHE",
              "completion_date": ""}
    with contextlib.redirect_stdout(io.StringIO()):
        dpl.writers(crn, out, "crn", header=header, column="std")
        back = dpl.read_crn(out + ".crn")
    assert list(back.columns) == ["std", "samp_depth"]
    j = crn.join(back, lsuffix="_o", rsuffix="_b")
    # writer stores round(index, 3) * 1000, so read-back == round(orig, 3) exactly
    assert np.nanmax(np.abs(j["std_o"].round(3) - j["std_b"])) < 1e-9
    assert np.nanmax(np.abs(j["samp_depth_o"] - j["samp_depth_b"])) == 0


# --- errors ------------------------------------------------------------------
def test_no_data_raises_or_warns(tmp_path):
    p = tmp_path / "empty.crn"
    p.write_text("just some prose\nno chronology here\n")
    with pytest.raises(ValueError):
        dpl.read_crn(str(p))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dpl.read_crn(str(p), strict=False) is None


def test_bad_strict_raises(tmp_path):
    p = tmp_path / "x.crn"
    p.write_text("\n".join(_HDR + [_row("SITEAB", 1900, [(1000, 4)])]) + "\n")
    with pytest.raises(ValueError):
        dpl.read_crn(str(p), strict="bogus")


# --- adaptive decade column (BC chronologies), à la dplR ----------------------
def _bc_row(site5, year, pairs):
    """A BC-convention data row: 5-char ID, then a signed 5-col year (the minus
    lives in column 6), then the (I4,I3) value pairs from column 11 -- the
    'MWK51-6000' layout of the Methuselah Walk bristlecone chronology."""
    cells = "".join("%4d%3d" % (v, d) for v, d in pairs)
    return ("%-5s" % site5) + ("%5d" % year) + cells


def test_bc_year_decade_shift_reads_one_series(tmp_path):
    # cols 7-10 of the first data line read as a future year (6000); dplPy must
    # detect the trailing negative and shift the decade field to col 6, reading a
    # single MWK51 series spanning BC to AD -- not splitting it into 'MWK51-'/'MWK51'.
    lines = [
        _bc_row("MWK51", -6000, [(1345, 11), (1077, 11), (1545, 11)]),
        _bc_row("MWK51", -5990, [(1000, 12), (900, 12)]),
    ]
    with pytest.warns(UserWarning, match="decade field"):
        df = _read(tmp_path, "bc.crn", lines)
    assert list(df.columns) == ["std", "samp_depth"]       # one series, not split
    assert int(df.index.min()) == -6000
    assert df.loc[-6000, "std"] == pytest.approx(1.345)
    assert df.loc[-5990, "std"] == pytest.approx(1.000)


def test_standard_file_keeps_decade_at_col_7(tmp_path):
    # A normal AD file must NOT trigger the shift (no spurious warning, decade at 7).
    lines = _HDR + [_row("SITEAB", 1900, [(1000, 5), (1100, 6)])]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # any warning would fail
        df = _read(tmp_path, "std.crn", lines)
    assert list(df.columns) == ["std", "samp_depth"]
    assert df.loc[1900, "std"] == pytest.approx(1.0)


# --- graceful failure on a non-fixed-width file (#3) --------------------------
def test_non_fixedwidth_file_fails_cleanly(tmp_path):
    # A whitespace-delimited file parses a "year" but no values; must fail cleanly
    # (clear message / None), never crash on an empty year set.
    p = tmp_path / "ws.crn"
    p.write_text("SITEAB 1900 1000 5 1100 6 950 7\n")
    with pytest.raises(ValueError, match="No chronology data"):
        with contextlib.redirect_stdout(io.StringIO()):
            dpl.read_crn(str(p))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dpl.read_crn(str(p), strict=False) is None


# --- all-1 sample depth dropped (dplR convention) -----------------------------
def test_all_one_sample_depth_is_dropped(tmp_path):
    # dplR convention: if every recorded sample depth is 1, the sample size was
    # never really recorded, so drop the misleading samp_depth column. Padding 0s
    # (from 9990 cells) are ignored, not counted as a non-1 depth.
    lines = _HDR + [_row("SITEAB", 1900, [(1000, 1), (1100, 1), (950, 1)]),
                    _row("SITEAB", 1910, [(1005, 1), (1002, 1)])]
    with pytest.warns(UserWarning, match="all recorded sample depths are 1"):
        df = _read(tmp_path, "one.crn", lines)
    assert list(df.columns) == ["std"]                     # samp_depth dropped
    assert df.loc[1900, "std"] == pytest.approx(1.0)


def test_all_one_depth_dropped_per_column(tmp_path):
    # In a multi-site file, only the all-1 site's depth column is dropped; a site
    # with real depths keeps its column.
    lines = [_row("SITEA", 1900, [(1000, 1), (1100, 1)]),
             _row("SITEB", 1900, [(900, 5), (950, 6)])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _read(tmp_path, "mix.crn", lines)
    assert "SITEA samp_depth" not in df.columns
    assert "SITEB samp_depth" in df.columns
    assert "SITEA" in df.columns and "SITEB" in df.columns  # values kept for both


def test_real_depths_are_not_dropped(tmp_path):
    # A depth column with any recorded value != 1 is kept.
    lines = _HDR + [_row("SITEAB", 1900, [(1000, 4), (1010, 4)])]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # no drop warning expected
        df = _read(tmp_path, "keep.crn", lines)
    assert "samp_depth" in df.columns


# --- wide multi-site bundle builds in one shot (#2) ---------------------------
def test_wide_multisite_no_fragmentation_warning(tmp_path):
    # _assemble must construct the frame at once: a wide bundle (120 columns) must
    # not raise pandas' "highly fragmented DataFrame" PerformanceWarning.
    rows = [_row("SIT%03d" % i, 1900, [(1000, 5), (1010, 5)]) for i in range(60)]
    p = tmp_path / "wide.crn"
    p.write_text("\n".join(rows) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        with contextlib.redirect_stdout(io.StringIO()):
            df = dpl.read_crn(str(p))
    assert df.shape[1] == 120                              # 60 sites x (value + depth)
