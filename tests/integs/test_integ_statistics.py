import io
import contextlib

import dplpy as dpl


def _ca533():
    return dpl.readers("./tests/data/csv/ca533.csv")


def test_stats_shape():
    st = dpl.stats(_ca533())
    assert list(st.columns) == ["series", "first", "last", "year", "mean",
                                "median", "stdev", "skew", "kurtosis", "gini", "ar1"]
    assert len(st) == 34                                  # one row per series


def test_summary_shape():
    sm = dpl.summary(_ca533())
    assert sm.shape == (8, 34)                            # describe() x 34 series
    assert list(sm.index[:3]) == ["count", "mean", "std"]


def test_report_prints_key_lines():
    # report() prints (returns None); capture stdout and check the headline lines
    # are present and populated (the intercorrelation line used to print empty).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dpl.report(_ca533())
    out = buf.getvalue()
    assert "Number of dated series: 34" in out
    assert "Mean (Std dev) series intercorrelation:" in out
    # the intercorrelation line now carries a number, not just the label
    ic_line = [l for l in out.splitlines() if "series intercorrelation" in l][0]
    assert any(ch.isdigit() for ch in ic_line.split(":", 1)[1])
