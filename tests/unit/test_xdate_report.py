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


def test_xdate_report_screen_only_writes_no_file(tmp_path):
    # output="screen" prints/returns the report but writes no file
    res = _quiet_report(RWL + "ca533.rwl", out_dir=str(tmp_path), output="screen")
    assert "text" in res[RWL + "ca533.rwl"]
    assert not os.listdir(str(tmp_path))          # nothing written


def test_xdate_report_bad_output_raises():
    with pytest.raises(ValueError):
        dpl.xdate_report(RWL + "ca533.rwl", output="bogus")


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
    cof = _quiet_report(key, out_dir=str(tmp_path), output="screen", preset="COFECHA")
    dfl = _quiet_report(key, out_dir=str(tmp_path), output="screen")
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


def test_xdate_report_accepts_dataframe(tmp_path):
    # An already-loaded DataFrame is reported directly (the read step is skipped),
    # treated as ONE collection -- not iterated into its columns (the bug that
    # produced "[i/289]" and "No such file: '644011'"). Result matches reading the
    # same file by path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            frame = dpl.readers(RWL + "ca533.rwl")
    res = _quiet_report(frame, out_dir=str(tmp_path), name="mem")
    assert "mem" in res and "text" in res["mem"]
    rep = res["mem"]["report"]
    assert rep["n_series"] == frame.shape[1]        # one collection, not 289 "files"
    assert len(rep["rows"]) == rep["n_series"]
    # a named DataFrame report writes <name>.txt
    assert os.path.exists(os.path.join(str(tmp_path), "mem.txt"))
    # same series count as reading the file by path
    by_path = _quiet_report(RWL + "ca533.rwl", out_dir=str(tmp_path), output="screen")
    assert rep["n_series"] == by_path[RWL + "ca533.rwl"]["report"]["n_series"]


def test_xdate_report_dataframe_default_name(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            frame = dpl.readers(RWL + "ca533.rwl")
    res = _quiet_report(frame, out_dir=str(tmp_path), output="screen")
    assert "report" in res                           # default label when no name given


# --- v0.9.0 audit follow-ups ---------------------------------------------------

def test_xdate_report_handles_bc_years(tmp_path):
    # B3: nm580 (first year 136 BC) crashed PART 5 on int("") from splitting the bin
    # label on "-"; bins are parsed with xdate._bin_bounds now.
    key = RWL + "nm580.rwl"
    for preset in (None, "COFECHA"):
        res = _quiet_report(key, out_dir=str(tmp_path), output="screen", preset=preset)
        assert "error" not in res[key], res[key].get("error")
        txt = res[key]["text"]
        assert "PART 5" in txt and "PART 7" in txt
        rep = res[key]["report"]
        assert rep["first_year"] < 0
        assert any(b.startswith("-") for b in rep["bins"])


def test_xdate_report_window_alias_and_case(tmp_path):
    # D3/D11/D13: slide_period= is a deprecated alias; preset/output are
    # case-insensitive; xdate is called with plot=False (no DeprecationWarning
    # from xdate's own make_plot= shim).
    key = RWL + "ca533.rwl"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            old = dpl.xdate_report(key, output="Screen", verbose=False, slide_period=40)
    dep = [str(m.message) for m in w if isinstance(m.message, DeprecationWarning)]
    assert any("slide_period" in d for d in dep)
    assert not any("make_plot" in d for d in dep)
    new = _quiet_report(key, output="screen", verbose=False, window_length=40)
    assert old[key]["text"] == new[key]["text"]
    assert new[key]["report"]["window_length"] == 40
    cof = _quiet_report(key, output="screen", verbose=False, preset="cofecha")
    assert cof[key]["report"]["preset"] == "COFECHA"
    with pytest.raises(TypeError):
        dpl.xdate_report(key, bogus=1)


def test_xdate_report_ar_order_failure_warns_not_zero():
    # G4: a series the AR fit cannot handle is reported (warning) and blank, not
    # silently shown as order 0.
    from dplpy.xdate_report import _ar_order
    import numpy as np
    import pandas as pd
    assert _ar_order(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])) == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _ar_order(pd.Series(["a", "b", "c", "d"]))
    assert np.isnan(out)
    assert any("AR order could not be determined" in str(m.message) for m in w)
