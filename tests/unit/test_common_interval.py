import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _read_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


# dplR common.interval() reference dimensions (validated against dplR 1.7.9):
#   (n_series, n_years, first_year, last_year)
_DPLR_REF = {
    ("co021", "series"): (33, 288, 1660, 1947),
    ("co021", "years"):  (27, 458, 1490, 1947),
    ("co021", "both"):   (28, 435, 1528, 1962),
    ("ca533", "series"): (30, 242, 1727, 1968),
    ("ca533", "years"):  (16, 722, 1247, 1968),
    ("ca533", "both"):   (22, 500, 1471, 1970),
}


@pytest.mark.parametrize("dataset,ctype", list(_DPLR_REF.keys()))
def test_common_interval_matches_dplR(dataset, ctype):
    data = _read_quiet("tests/data/csv/%s.csv" % dataset)
    ci = dpl.common_interval(data, type=ctype)
    n_series, n_years, y0, y1 = _DPLR_REF[(dataset, ctype)]
    assert ci.shape == (n_years, n_series)
    assert int(ci.index.min()) == y0 and int(ci.index.max()) == y1
    # the result is a complete rectangle: no missing values anywhere
    assert not ci.isna().any().any()


def test_common_interval_bad_type():
    data = _read_quiet("tests/data/csv/co021.csv")
    with pytest.raises(ValueError):
        dpl.common_interval(data, type="bogus")


def test_common_interval_bad_input():
    with pytest.raises(TypeError):
        dpl.common_interval("not a frame")


def test_common_interval_rescues_empty_common_period():
    # On unevenly-distributed co021 the strict all-series intersection is empty,
    # so rwi_stats(period='common') is NaN. A common_interval selection restores
    # a valid, complete block and hence a real rbar/EPS.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(_read_quiet("tests/data/csv/co021.csv"),
                              fit="spline", plot=False)
            empty = dpl.rwi_stats(rwi, period="common")
            both = dpl.rwi_stats(rwi, common_interval="both")
            span = dpl.rwi_stats(rwi, common_interval=(1600, 1900))
    assert np.isnan(empty["rbar_eff"][0])                 # the original problem
    assert both["n_cores"][0] == 28                       # dplR 'both' -> 28 series
    assert np.isfinite(both["rbar_eff"][0])
    # user-specified span keeps only series with data across it
    assert span["n_cores"][0] >= 2 and np.isfinite(span["rbar_eff"][0])


def test_sss_common_interval_matches_dplR_composition():
    # dplR's recipe for a common-interval SSS is sss(common.interval(rwi)); the
    # sss(common_interval=...) convenience must reproduce that exactly, and SSS
    # must stay <= 1 across the trimmed block.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(_read_quiet("tests/data/csv/co021.csv"),
                              fit="spline", plot=False)
            via_param = dpl.sss(rwi, common_interval="both")
            via_compose = dpl.sss(dpl.common_interval(rwi, type="both"))
            full = dpl.sss(rwi)
    assert list(via_param.index) == list(via_compose.index)
    assert np.allclose(via_param.to_numpy(), via_compose.to_numpy())
    # trimmed to the 'both' interval (1528-1962), and SSS <= 1 there
    assert int(via_param.index.min()) == 1528 and int(via_param.index.max()) == 1962
    assert (via_param <= 1 + 1e-9).all()
    # the no-interval default still spans the full record (dplR behaviour)
    assert int(full.index.min()) == 1176 and int(full.index.max()) == 1963


def test_rwi_stats_bad_common_interval():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(_read_quiet("tests/data/csv/co021.csv"),
                              fit="spline", plot=False)
    with pytest.raises(ValueError):
        dpl.rwi_stats(rwi, common_interval="nonsense")


# --- v0.9.0 audit follow-ups ------------------------------------------------

def _rm_short_reference(notNA, row_idx, flag):
    """The pre-0.9.0 rm.short (full re-test of every candidate row after each
    drop, O(nrow x ncol) per step) kept as the reference for F4."""
    sub = notNA[row_idx, :]
    m, ncol_full = sub.shape
    which_good = np.where(sub.any(axis=0))[0]
    ncol_orig = len(which_good)
    span_len = np.empty(ncol_orig)
    for k, c in enumerate(which_good):
        pos = np.where(sub[:, c])[0]
        span_len[k] = pos[-1] - pos[0]
    span_order = which_good[np.argsort(span_len, kind="stable")]
    keep_col = np.zeros(ncol_full, dtype=bool)
    keep_col[which_good] = True
    keep_col_out = keep_col.copy()
    dontkeep_row = np.ones(m, dtype=bool)
    keep_row_out = np.zeros(m, dtype=bool)
    nrow = nrow_out = 0
    ncol_out = ncol = ncol_orig
    best = 0
    for i in range(0, max(0, ncol_orig - 2) + 1):
        if i > 0:
            keep_col[span_order[i - 1]] = False
            ncol -= 1
            if ncol * m < best:
                break
        cand = np.where(dontkeep_row)[0]
        tmp = (sub[np.ix_(cand, np.where(keep_col)[0])].all(axis=1)
               if len(cand) else np.zeros(0, dtype=bool))
        dontkeep_row[cand] = ~tmp
        nrow += int(tmp.sum())
        area = ncol * nrow
        if area > best:
            best = area
            keep_col_out = keep_col.copy()
            keep_row_out = ~dontkeep_row.copy()
            ncol_out = ncol
            nrow_out = nrow
            if flag:
                break
    return nrow_out, ncol_out, keep_row_out, keep_col_out


def _check_rm_short_equal(notNA):
    from dplpy.common_interval import _rm_short
    depth = notNA.sum(axis=1)
    for i in range(int(depth.max()), 1, -1):
        rows = np.where(depth >= i)[0]
        if len(rows) == 0:
            continue
        row_idx = np.arange(rows[0], rows[-1] + 1)
        for flag in (True, False):
            a = _rm_short(notNA, row_idx, flag)
            b = _rm_short_reference(notNA, row_idx, flag)
            assert a[0] == b[0] and a[1] == b[1]
            assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])


def test_rm_short_running_count_matches_reference_ca533():
    data = _read_quiet("tests/data/csv/ca533.csv")
    _check_rm_short_equal(~data.isna().to_numpy())


def test_rm_short_running_count_matches_reference_synthetic():
    rng = np.random.default_rng(7)
    nrow, ncol = 120, 25
    notNA = np.zeros((nrow, ncol), dtype=bool)
    for j in range(ncol):
        a = rng.integers(0, nrow - 10)
        b = rng.integers(a + 5, nrow)
        notNA[a:b, j] = True
        # a few interior gaps
        for g in rng.integers(a, b, size=rng.integers(0, 3)):
            notNA[g, j] = False
    _check_rm_short_equal(notNA)
    df = pd.DataFrame(np.where(notNA, 1.0, np.nan), index=pd.Index(range(nrow), name="Year"),
                      columns=["s%d" % j for j in range(ncol)])
    for t in ("series", "years", "both"):
        ci = dpl.common_interval(df, type=t)
        assert not ci.isna().any().any()                 # a complete rectangle


def test_common_interval_type_case_insensitive_and_attrs():
    data = _read_quiet("tests/data/csv/ca533.csv")
    data.attrs.update({"dplpy_precision": 0.01, "kind": "rwl", "site": "ca533"})
    a = dpl.common_interval(data, type="Years")
    b = dpl.common_interval(data, type="years")
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["kind"] == "rwl" and a.attrs["dplpy_precision"] == 0.01
    assert a.attrs["site"] == "ca533"
