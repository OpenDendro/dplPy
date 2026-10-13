# Tests for dpl.check(): the findings-as-data integrity checker.
# Clean data stays quiet; each planted defect raises its own finding; the
# engine never stops on a failing check; file-level and summary paths work.

import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _clean(nseries=12, seed=11):
    """A synthetic collection with a strong year-to-year common signal, a
    positive biological growth curve, and no defects. Nothing should fire."""
    rng = np.random.default_rng(seed)
    years = np.arange(1700, 1991)
    n = len(years)
    common = rng.normal(0, 1, n)
    data = {}
    for i in range(nseries):
        age = np.arange(1, n + 1)
        trend = 3.0 * np.exp(-age / 70.0) + 0.3
        rwi = 1.0 + 0.35 * common + rng.normal(0, 0.12, n)
        data["T%02d" % (i + 1)] = np.round(np.clip(trend * rwi, 0.02, None), 3)
    return pd.DataFrame(data, index=pd.Index(years, name="Year"))


def test_clean_collection_is_quiet():
    f = dpl.check(_clean(), file="clean")
    assert list(f.columns) == ["file", "check", "severity", "group", "series",
                               "year_from", "year_to", "n", "value", "message"]
    # a clean synthetic draws no errors, no warnings, and no crossdating flags
    assert (f["severity"] == "error").sum() == 0
    assert (f["severity"] == "warning").sum() == 0
    assert (f["group"] == "crossdating").sum() == 0


def test_repeated_value_sentinel_is_caught():
    # the real-world defect: a constant placeholder (here 1.000) used where a
    # ring was not measured. It is not negative and not zero, so the repeated
    # run is what exposes it.
    df = _clean()
    df.iloc[0:25, df.columns.get_loc("T04")] = 1.000
    f = dpl.check(df)
    hit = f[(f["check"] == "RWL_REPEATED_VALUE") & (f["series"] == "T04")]
    assert len(hit) == 1
    assert hit.iloc[0]["value"] == 1.0
    assert hit.iloc[0]["n"] == 25


def test_zero_variance_series():
    df = _clean()
    df["FLAT"] = 1.5
    f = dpl.check(df)
    assert ((f["check"] == "RWL_ZERO_VARIANCE") & (f["series"] == "FLAT")).any()


def test_duplicate_series():
    df = _clean()
    df["DUP1"] = df["T01"].values
    f = dpl.check(df)
    assert (f["check"] == "RWL_DUP_SERIES").any()


def test_short_series():
    df = _clean()
    short = pd.Series(np.nan, index=df.index)
    short.iloc[0:12] = df["T02"].values[:12]
    df["SHORT"] = short
    f = dpl.check(df)
    hit = f[(f["check"] == "RWL_SHORT_SERIES") & (f["series"] == "SHORT")]
    assert len(hit) == 1
    assert hit.iloc[0]["n"] == 12


def test_negative_values():
    df = _clean()
    df.iloc[0:10, df.columns.get_loc("T02")] = -9.0
    f = dpl.check(df)
    hit = f[f["check"] == "RWL_NEGATIVE"]
    assert len(hit) == 1
    assert hit.iloc[0]["value"] < 0


def test_internal_na():
    df = _clean()
    df.iloc[150, df.columns.get_loc("T03")] = np.nan
    f = dpl.check(df)
    assert ((f["check"] == "RWL_INTERNAL_NA") & (f["series"] == "T03")).any()


def test_dating_lag_on_whole_series_offset():
    # a series dated one year off correlates best with the master at a non-zero
    # lag. dplR sign convention: a negative best lag flags a missing ring.
    df = _clean()
    s = df["T05"].copy()
    s.index = s.index + 1
    df["T05"] = s.reindex(df.index)
    f = dpl.check(df)
    hit = f[(f["check"] == "RWL_DATING_LAG") & (f["series"] == "T05")]
    assert len(hit) == 1
    assert int(hit.iloc[0]["value"]) != 0


def test_summary_mode_one_row_per_file():
    df = _clean()
    df["FLAT"] = 1.5
    s = dpl.check(df, file="one", summary=True)
    assert len(s) == 1
    assert s.iloc[0]["file"] == "one"
    assert s.iloc[0]["n_error"] >= 1          # the constant series is an error
    assert "RWL_ZERO_VARIANCE" in s.columns


def test_check_group_selection():
    df = _clean()
    df["FLAT"] = 1.5
    only_series = dpl.check(df, checks=["series"])
    assert set(only_series["group"].unique()) <= {"series"}
    assert (only_series["check"] == "RWL_ZERO_VARIANCE").any()


def test_unknown_check_group_raises():
    with pytest.raises(ValueError):
        dpl.check(_clean(), checks=["not_a_group"])


def test_file_group_on_crafted_bytes(tmp_path):
    p = tmp_path / "defect.rwl"
    body = (b"FOO01 1950   120   130\t140   150\r\n"   # tab + CRLF
            b"FOO01 1954   160   170   180   190\n"     # LF
            b"BAR\xe9  1950   110   120   130   140\n"   # non-ASCII byte
            b"BAR01 1954   150   160   170   180")       # no final newline
    p.write_bytes(body)
    f = dpl.check(str(p), checks=["file"])
    got = set(f["check"])
    assert "RWL_TAB" in got
    assert "RWL_MIXED_EOL" in got
    assert "RWL_NON_ASCII" in got
    assert "RWL_NO_FINAL_NEWLINE" in got


def test_reads_a_path_and_runs():
    # a real file reads and checks without raising; ca533 is well-formed, so it
    # draws no errors (its zero rings are locally absent and reported as notes).
    import os
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "rwl", "ca533.rwl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = dpl.check(path)
    assert (f["severity"] == "error").sum() == 0
    assert isinstance(f, pd.DataFrame)
