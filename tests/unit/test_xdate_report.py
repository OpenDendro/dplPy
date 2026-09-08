import io
import contextlib
import warnings
import os

import pytest

import dplpy as dpl

RWL = "tests/data/rwl/"


def _quiet_report(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return dpl.xdate_report(*args, **kw)


def test_xdate_report_sections_and_structure(tmp_path):
    # A real multi-series file produces a COFECHA-style report with the header block
    # and PART 5 / PART 7 sections, and writes one .txt per input.
    res = _quiet_report(RWL + "ca533.rwl", out_dir=str(tmp_path))
    key = RWL + "ca533.rwl"
    assert key in res and "text" in res[key]
    txt = res[key]["text"]
    assert "Series intercorrelation" in txt
    assert "PART 5:  CORRELATION OF SERIES BY SEGMENTS" in txt
    assert "PART 7:  DESCRIPTIVE STATISTICS" in txt
    assert "Number problem segments" in txt
    # one report file written, named by the input stem
    assert os.path.exists(os.path.join(str(tmp_path), "ca533.txt"))
    # per-series rows are present in the report dict
    rep = res[key]["report"]
    assert rep["n_series"] > 1 and len(rep["rows"]) == rep["n_series"]
    r0 = rep["rows"][0]
    for col in ("first", "last", "nyears", "corr", "mean", "sens", "ar"):
        assert col in r0


def test_xdate_report_write_false_returns_text_only(tmp_path):
    res = _quiet_report(RWL + "ca533.rwl", out_dir=str(tmp_path), write=False)
    assert "text" in res[RWL + "ca533.rwl"]
    assert not os.listdir(str(tmp_path))          # nothing written


def test_xdate_report_bad_file_is_recorded_not_raised(tmp_path):
    # A batch must not derail on one unreadable file: it is recorded as an error.
    bad = tmp_path / "notrwl.rwl"
    bad.write_text("this is not tree-ring data\njust prose\n")
    res = _quiet_report([str(bad)], out_dir=str(tmp_path))
    assert "error" in res[str(bad)]
