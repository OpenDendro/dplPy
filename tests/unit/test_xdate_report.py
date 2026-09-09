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


def test_xdate_report_cofecha_preset(tmp_path):
    # preset="COFECHA" routes the batch report through xdate's COFECHA emulation
    # (Burg / variance stabilization / omit-absent / length-weighted summary) and
    # still produces a well-formed report. Its summary differs from the default.
    key = RWL + "ca533.rwl"
    cof = _quiet_report(key, out_dir=str(tmp_path), write=False, preset="COFECHA")
    dfl = _quiet_report(key, out_dir=str(tmp_path), write=False)
    assert cof[key]["report"]["preset"] == "COFECHA"
    assert dfl[key]["report"].get("preset") is None
    ctxt, dtxt = cof[key]["text"], dfl[key]["text"]
    assert "Series intercorrelation" in ctxt and "PART 5" in ctxt and "PART 7" in ctxt
    # the two presets give different summary numbers (different transform)
    assert ctxt != dtxt

    # length-weighted intercorrelation in COFECHA mode reproduces the COFECHA run
    # on ca533 (0.668) far better than the plain mean the default reports.
    import re
    def _intercorr(t):
        return float(re.search(r"Series intercorrelation:\s*([0-9.]+)", t).group(1))
    assert abs(_intercorr(ctxt) - 0.668) < 0.01
