"""Non-mutation sweep: no public data-transforming function may modify the
DataFrame (or Series) it is given.

Every case snapshots its input(s) (values, index, columns, dtypes and
``.attrs``) before the call and asserts they are identical afterwards. A function
that needs to change the data must work on a copy and return it.
"""
import copy
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


# ----------------------------------------------------------------- helpers ---

def _snapshot(obj):
    """Deep snapshot of a DataFrame/Series: data + attrs."""
    return obj.copy(deep=True), copy.deepcopy(dict(obj.attrs))


def _assert_unchanged(obj, snap, label):
    before, attrs = snap
    if isinstance(obj, pd.DataFrame):
        pd.testing.assert_frame_equal(obj, before, check_exact=True,
                                      check_names=True, obj=label)
        assert list(obj.columns) == list(before.columns), label + ": columns"
    else:
        pd.testing.assert_series_equal(obj, before, check_exact=True,
                                       check_names=True, obj=label)
    assert obj.index.equals(before.index), label + ": index"
    assert obj.index.name == before.index.name, label + ": index name"
    assert dict(obj.attrs) == attrs, label + ": attrs changed"


def _call(fn, *args, **kwargs):
    """Call fn, passing only the keyword arguments its signature names
    explicitly (so plot=/verbose= are given where they exist and silently
    dropped where they do not)."""
    params = inspect.signature(fn).parameters
    kw = {k: v for k, v in kwargs.items() if k in params}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


QUIET = dict(plot=False, verbose=False, show_flags=False)


# ------------------------------------------------------------------- cases ---
# Each case builder takes (rwl, rwi) -- fresh copies of ca533 -- and returns
# (inputs_to_watch, thunk). ``inputs_to_watch`` lists every pandas object the
# thunk passes in; all of them must come back unchanged.

def _c_detrend(rwl, rwi):
    return [rwl], lambda: _call(dpl.detrend, rwl, **QUIET)


def _c_fill_internal(rwl, rwi):
    gappy = rwl.copy()
    col = gappy.columns[0]
    span = gappy[col].dropna().index
    gappy.loc[span[len(span) // 2], col] = np.nan            # one interior gap
    return [gappy], lambda: _call(dpl.fill_internal, gappy, **QUIET)


def _c_edit_rwl(rwl, rwi):
    col = rwl.columns[0]
    year = int(rwl[col].dropna().index[10])
    return [rwl], lambda: _call(dpl.edit_rwl, rwl, series=col, year=year,
                                action="insert", value=0.5)


def _c_combine_rwl(rwl, rwi):
    a, b = rwl.iloc[:, :10].copy(), rwl.iloc[:, 10:].copy()
    return [a, b], lambda: _call(dpl.combine_rwl, a, b)


def _c_powt(rwl, rwi):
    return [rwl], lambda: _call(dpl.powt, rwl, **QUIET)


def _c_bai_out(rwl, rwi):
    return [rwl], lambda: _call(dpl.bai_out, rwl)


def _c_bai_in(rwl, rwi):
    return [rwl], lambda: _call(dpl.bai_in, rwl)


def _c_tree_mean(rwl, rwi):
    ids = _call(dpl.read_ids, rwl)
    return [rwl, ids], lambda: _call(dpl.tree_mean, rwl, ids)


def _c_rcs(rwl, rwi):
    return [rwl], lambda: _call(dpl.rcs, rwl, **QUIET)


def _c_sfrcs(rwl, rwi):
    return [rwl], lambda: _call(dpl.sfrcs, rwl, **QUIET)


def _c_mrcs(rwl, rwi):
    return [rwl], lambda: _call(dpl.mrcs, rwl, **QUIET)


def _c_sfmrcs(rwl, rwi):
    return [rwl], lambda: _call(dpl.sfmrcs, rwl, **QUIET)


def _c_ssf_recode_zeros(rwl, rwi):
    assert (rwl == 0).to_numpy().sum() > 0                   # ca533 has zeros
    return [rwl], lambda: _call(dpl.ssf, rwl, recode_zeros=True, **QUIET)


def _c_chron(rwl, rwi):
    return [rwi], lambda: _call(dpl.chron, rwi, **QUIET)


def _c_chron_prewhiten_ci(rwl, rwi):
    return [rwi], lambda: _call(dpl.chron, rwi, prewhiten=True, ci=True, R=5,
                                seed=1, **QUIET)


def _c_chron_stabilized(rwl, rwi):
    return [rwi], lambda: _call(dpl.chron_stabilized, rwi, **QUIET)


def _c_chron_ars(rwl, rwi):
    return [rwi], lambda: _call(dpl.chron_ars, rwi, **QUIET)


def _c_stabilize_chron_rbar(rwl, rwi):
    crn = _call(dpl.chron, rwi, **QUIET)
    return [crn, rwi], lambda: _call(dpl.stabilize_chron, crn, "rbar", rwi=rwi)


def _c_stabilize_chron_spline(rwl, rwi):
    crn = _call(dpl.chron, rwi, **QUIET)
    return [crn], lambda: _call(dpl.stabilize_chron, crn, "spline")


def _c_ar_func(rwl, rwi):
    return [rwl], lambda: _call(dpl.ar_func, rwl)


def _c_ar_func_series(rwl, rwi):
    s = rwl[rwl.columns[0]].dropna().copy()
    return [s], lambda: _call(dpl.ar_func, s)


def _c_xdate(rwl, rwi):
    return [rwi], lambda: _call(dpl.xdate, rwi, **QUIET)


def _c_series_corr(rwl, rwi):
    return [rwi], lambda: _call(dpl.series_corr, rwi, rwi.columns[0], **QUIET)


def _c_interseries_corr(rwl, rwi):
    return [rwl], lambda: _call(dpl.interseries_corr, rwl)


def _c_rwi_stats(rwl, rwi):
    return [rwi], lambda: _call(dpl.rwi_stats, rwi)


def _c_rwi_stats_prewhiten(rwl, rwi):
    return [rwi], lambda: _call(dpl.rwi_stats, rwi, prewhiten=True)


def _c_rwi_stats_running(rwl, rwi):
    return [rwi], lambda: _call(dpl.rwi_stats_running, rwi)


def _c_sss(rwl, rwi):
    return [rwi], lambda: _call(dpl.sss, rwi)


def _c_glk(rwl, rwi):
    return [rwl], lambda: _call(dpl.glk, rwl)


def _c_sens1(rwl, rwi):
    return [rwl], lambda: _call(dpl.sens1, rwl)


def _c_common_interval(rwl, rwi):
    return [rwl], lambda: _call(dpl.common_interval, rwl, **QUIET)


def _c_check(rwl, rwi):
    return [rwl], lambda: _call(dpl.check, rwl)


def _c_summary(rwl, rwi):
    return [rwl], lambda: _call(dpl.summary, rwl)


def _c_stats(rwl, rwi):
    return [rwl], lambda: _call(dpl.stats, rwl)


def _c_report(rwl, rwi):
    return [rwl], lambda: _call(dpl.report, rwl)


def _c_samp_stats(rwl, rwi):
    return [rwl], lambda: _call(dpl.samp_stats, rwl)


def _c_xdate_floater(rwl, rwi):
    col = rwl.columns[0]
    master = rwl.drop(columns=[col])
    floater = rwl[col].dropna().reset_index(drop=True)
    floater.name = "FLOAT"
    return [master, floater], lambda: _call(dpl.xdate_floater, master, floater,
                                            **QUIET)


def _c_xdate_undated(rwl, rwi):
    # a handful of long, overlapping series keeps the pairwise search quick
    counts = rwl.notna().sum().sort_values(ascending=False)
    small = rwl[counts.index[:4]].dropna(how="all")
    small = small.loc[small.index[-350:]].dropna(how="all")   # ~350 rings each
    return [small], lambda: _call(dpl.xdate_undated, small, **QUIET)


def _c_releases(rwl, rwi):
    return [rwl], lambda: _call(dpl.releases, rwl)


def _c_disturbance_detrend(rwl, rwi):
    return [rwl], lambda: _call(dpl.disturbance_detrend, rwl, **QUIET)


CASES = {
    "detrend": _c_detrend,
    "releases": _c_releases,
    "disturbance_detrend": _c_disturbance_detrend,
    "fill_internal": _c_fill_internal,
    "edit_rwl": _c_edit_rwl,
    "combine_rwl": _c_combine_rwl,
    "powt": _c_powt,
    "bai_out": _c_bai_out,
    "bai_in": _c_bai_in,
    "tree_mean": _c_tree_mean,
    "rcs": _c_rcs,
    "sfrcs": _c_sfrcs,
    "mrcs": _c_mrcs,
    "sfmrcs": _c_sfmrcs,
    "ssf(recode_zeros=True)": _c_ssf_recode_zeros,
    "chron": _c_chron,
    "chron(prewhiten,ci)": _c_chron_prewhiten_ci,
    "chron_stabilized": _c_chron_stabilized,
    "chron_ars": _c_chron_ars,
    "stabilize_chron(rbar)": _c_stabilize_chron_rbar,
    "stabilize_chron(spline)": _c_stabilize_chron_spline,
    "ar_func": _c_ar_func,
    "ar_func(Series)": _c_ar_func_series,
    "xdate": _c_xdate,
    "series_corr": _c_series_corr,
    "interseries_corr": _c_interseries_corr,
    "rwi_stats": _c_rwi_stats,
    "rwi_stats(prewhiten)": _c_rwi_stats_prewhiten,
    "rwi_stats_running": _c_rwi_stats_running,
    "sss": _c_sss,
    "glk": _c_glk,
    "sens1": _c_sens1,
    "common_interval": _c_common_interval,
    "check": _c_check,
    "summary": _c_summary,
    "stats": _c_stats,
    "report": _c_report,
    "samp_stats": _c_samp_stats,
    "xdate_floater": _c_xdate_floater,
    "xdate_undated": _c_xdate_undated,
}


@pytest.mark.parametrize("name", list(CASES))
def test_public_function_does_not_mutate_input(name, ca533_rwl, ca533_rwi, capsys):
    inputs, thunk = CASES[name](ca533_rwl, ca533_rwi)
    snaps = [_snapshot(x) for x in inputs]
    thunk()
    capsys.readouterr()                       # swallow console output
    for i, (obj, snap) in enumerate(zip(inputs, snaps)):
        _assert_unchanged(obj, snap, "%s input #%d" % (name, i))


def test_sweep_covers_every_public_transform():
    # keep the sweep honest: every public function that takes a frame and is
    # not a reader/writer/plot/CLI helper must have a case above.
    exempt = {
        # I/O, metadata, and non-frame helpers
        "readers", "readers_url", "metadata", "read_crn", "read_ids", "writers",
        "read_tridas", "write_tridas", "to_lipd", "from_lipd", "lipd_metadata",
        "SiteMetadata", "TridasProject", "TridasMetadata", "help", "readme",
        "read_pos", "PosCollection",
        # climate-response functions: driven by climate data, not the ca533
        # ring-width fixtures this sweep uses -- each carries its own
        # non-mutation test in its own module.
        "seascorr", "SeascorrResult",
        "pcreg", "PcregResult",
        # plotting only
        "plot", "xdate_plot", "disturbance_plot",
        # file-driven report
        "xdate_report",
        # standalone numerical helpers (take arrays/series or special tables)
        "ads", "autoreg", "po_to_wc", "wc_to_po", "yrs_to_pith", "sens2", "sgc",
    }
    covered = {n.split("(")[0] for n in CASES}
    missing = [n for n in dpl.__all__ if n not in exempt and n not in covered]
    assert not missing, "no-mutation case missing for: %s" % missing
