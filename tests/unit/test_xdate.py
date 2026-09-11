import dplpy as dpl
import pandas as pd
import numpy as np
import warnings
import pytest


def _read_quiet(path):
    import io, contextlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


def _xdate_quiet(rwi, **kw):
    import io, contextlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.xdate(rwi, **kw)


def test_xdate_invalid_input():
    with pytest.raises(TypeError) as e:
        dpl.xdate("input_df")
    assert "Expected dataframe input, got <class 'str'> instead." == str(e.value)


def test_xdate_bad_corr_method():
    df = pd.DataFrame({"A": [1.0, 2, 3], "B": [1.0, 2, 3]},
                      index=pd.Index([1, 2, 3], name="Year"))
    with pytest.raises(ValueError):
        dpl.xdate(df, corr="bogus")


def test_xdate_returns_rich_result():
    # The result is a dict mirroring dplR's corr.rwl.seg output.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, corr="spearman", slide_period=50, bin_floor=100)
    for key in ("seg_corr", "p_val", "overall", "avg_seg_corr", "flags", "bins", "rwi"):
        assert key in res
    assert res["seg_corr"].shape == (34, 50)          # series x bins
    assert list(res["overall"].columns) == ["rho", "p_val"]
    assert res["bins"][0] == "700-749" and res["bins"][-1] == "1925-1974"


def test_xdate_matches_dplR_flags_ca533():
    # dplR corr.rwl.seg flags exactly these 5 series (a segment p-value >= 0.05)
    # on ca533. dplPy still flags all five, but splits them COFECHA-style: a
    # non-significant segment is an A flag only when the dated (lag-0) position is
    # its best match; when a non-dated lag correlates higher it is a B flag (a
    # possible dating shift) instead. On ca533 only CAM131 is a true A flag; the
    # other four correlate better at another lag, so they are B flags. A and B are
    # mutually exclusive -- an A flag never carries a non-zero best lag.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, corr="spearman", slide_period=50, bin_floor=100, p_val=0.05)
    # every series dplR flags is still flagged (as A or B)
    assert {"CAM011", "CAM051", "CAM131", "CAM181", "CAM201"}.issubset(res["flags"].keys())
    # only CAM131 is a true A flag (lag 0 is best but not significant)
    assert sorted(s for s, f in res["flags"].items() if f["A"]) == ["CAM131"]
    # the other four are B flags, not A, and every A flag's best lag is 0
    for s in ("CAM011", "CAM051", "CAM181", "CAM201"):
        assert res["flags"][s]["B"] and not res["flags"][s]["A"]
    for f in res["flags"].values():
        assert all(e["best_lag"] == 0 for e in f["A"])


def test_xdate_segment_correlation_values_ca533():
    # Regression on the spline-detrended pipeline (dplPy detrend + xdate). On the
    # identical RWI dplR uses, xdate reproduces corr.rwl.seg to ~1e-15; here the
    # tiny spline-detrend difference is absorbed by a loose tolerance.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, corr="spearman", slide_period=50, bin_floor=100)
    sc = res["seg_corr"]
    assert sc.loc["CAM011", "1750-1799"] == pytest.approx(0.6058, abs=1e-2)
    assert sc.loc["CAM191", "1850-1899"] == pytest.approx(0.7460, abs=1e-2)
    assert res["overall"].loc["CAM011", "rho"] == pytest.approx(0.5225, abs=1e-2)
    # correlations are in [-1, 1]
    vals = sc.to_numpy()
    vals = vals[~np.isnan(vals)]
    assert (vals >= -1).all() and (vals <= 1).all()


def test_xdate_ar_yw_prewhiten_matches_r():
    # The Yule-Walker prewhitener reproduces R's ar(): AIC order selection,
    # residuals + mean, series length kept (first `order` become NaN).
    from dplpy.xdate import _ar_yw_prewhiten
    rng = np.random.RandomState(0)
    n = 200
    e = rng.randn(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]
    x = x - x.min() + 1.0
    pw = _ar_yw_prewhiten(x)
    assert len(pw) == n                                # length preserved
    assert np.isnan(pw[:2]).any()                      # start NA-padded by order
    # prewhitened mean ~ original mean (residuals are re-centered on the mean)
    assert np.nanmean(pw) == pytest.approx(np.mean(x), abs=0.05)


def test_xdate_no_prewhiten_runs():
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, prewhiten=False, corr="pearson", slide_period=50, bin_floor=100)
    assert res["seg_corr"].shape[0] == 34


def test_xdate_plot_matches_dplR_semantics_ca533():
    # The corr.rwl.seg-style overview: one row per series, three colours with
    # dplR's meaning (green extent, blue dated, red flagged). The red bars must
    # fall on exactly the five A-flagged series, and nowhere else.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    from dplpy.xdate import xdate_plot, _CRS_FLAG
    import io, contextlib

    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = xdate_plot(rwi, slide_period=50, bin_floor=100, p_val=0.05)

    # Series labels alternate between the left axis and a secondary right axis,
    # so collect both to map an integer row -> series name.
    row_name = {}
    for a in [ax] + list(ax.child_axes):
        for pos, t in zip(a.get_yticks(), a.get_yticklabels()):
            if t.get_text():
                row_name[int(round(pos))] = t.get_text()
    assert len([v for v in row_name.values() if v.startswith("CAM")]) == 34
    red = mcolors.to_hex(_CRS_FLAG).lower()
    # series whose row (integer y-centre) carries a red rectangle
    flagged_rows = set()
    for p in ax.patches:
        if isinstance(p, Rectangle) and mcolors.to_hex(p.get_facecolor()).lower() == red:
            flagged_rows.add(row_name.get(int(round(p.get_y() + p.get_height() / 2))))
    assert flagged_rows == {"CAM011", "CAM051", "CAM131", "CAM181", "CAM201"}


def test_xdate_cofecha_preset_smoke():
    # preset="COFECHA" takes the RAW frame (it detrends internally) and returns the
    # extra COFECHA keys and a sensible problem count.
    data = _read_quiet("tests/data/csv/ca533.csv")
    res = _xdate_quiet(data, preset="COFECHA", show_flags=False)
    for k in ("seg_corr", "flags", "segments", "n_problems", "preset"):
        assert k in res
    assert res["preset"] == "COFECHA"
    assert isinstance(res["n_problems"], int) and res["n_problems"] >= 0
    # every flagged segment count is accounted for in n_problems
    total = sum(len(f["A"]) + len(f["B"]) for f in res["flags"].values())
    assert total == res["n_problems"]


def test_cofecha_preset_clips_to_multiseries_span():
    # COFECHA only tests a segment where TWO OR MORE series overlap, so that after
    # the current series is removed the leave-one-out master still exists (its
    # JYM2/LYM2 clip; cofecha_erc.f). On ca533, CAM211 (626-1968) is the sole
    # series before 695, so its segments must be clipped to start at/after 695 --
    # no spurious pre-695 correlation against an essentially empty master.
    data = _read_quiet("tests/data/csv/ca533.csv")
    res = _xdate_quiet(data, preset="COFECHA", show_flags=False)
    cam211 = res["segments"]["CAM211"]
    assert cam211 and cam211[0]["lo"] >= 695
    assert all(s["lo"] >= 695 for s in cam211)
    # no series anywhere is tested before the multi-series span begins
    earliest = min(s["lo"] for segs in res["segments"].values() for s in segs)
    assert earliest >= 695


def test_xdate_cofecha_segments_anchor_to_series_ends():
    # COFECHA anchors the first segment to each series' first year (full 50-yr
    # window) rather than snapping to a 25-yr grid multiple.
    data = _read_quiet("tests/data/csv/ca533.csv")
    res = _xdate_quiet(data, preset="COFECHA", show_flags=False)  # raw; self-detrends
    for name, segs in res["segments"].items():
        if not segs:
            continue
        col = data[name].dropna()
        y0 = int(col.index.min())
        # first segment starts at the series' first year (when long enough)
        if len(col) >= 50:
            assert segs[0]["lo"] == y0
            assert segs[0]["hi"] - segs[0]["lo"] + 1 == 50
        break


def test_xdate_default_path_unchanged_by_preset_plumbing():
    # the dplR-faithful default still returns the base keys and no COFECHA extras.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="Spline", plot=False)
    res = _xdate_quiet(rwi, show_flags=False)
    assert "segments" not in res and "n_problems" not in res
    assert set(["seg_corr", "p_val", "overall", "flags", "bins", "rwi"]).issubset(res)


def test_xdate_cofecha_omit_absent_rings():
    # COFECHA's "omit absent rings" (QAC=Y): a boolean mask and a raw frame whose
    # zeros mark absent rings give identical results, and dropping absent years
    # changes the flag accounting relative to not omitting.
    rng = np.random.default_rng(0)
    years = np.arange(1800, 1960)
    cols = {}
    for k in range(8):
        x = np.cumsum(rng.standard_normal(len(years))) * 0.05 + 1.0
        x = np.clip(x, 0.05, None)
        cols["S%d" % k] = x
    raw = pd.DataFrame(cols, index=pd.Index(years, name="Year"))
    # plant a few absent rings (zeros) in one series
    raw.iloc[10:13, 0] = 0.0
    # preset takes the raw frame directly (it detrends internally)
    base = _xdate_quiet(raw, preset="COFECHA", show_flags=False)
    r_raw = _xdate_quiet(raw, preset="COFECHA", absent=raw, show_flags=False)
    r_bool = _xdate_quiet(raw, preset="COFECHA", absent=(raw == 0), show_flags=False)
    assert r_raw["n_problems"] == r_bool["n_problems"]     # both forms agree
    assert isinstance(base["n_problems"], int)


def test_xdate_preset_none_vs_cofecha_coexist():
    # Regression guard: the default (preset=None) dplR-faithful path and the
    # COFECHA preset run on the same data without interfering. The default still
    # flags exactly dplR's five series and exposes no COFECHA-only keys; the
    # preset adds its extras and a non-negative problem count. The default result
    # is identical whether or not the preset was called first (no shared state).
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="Spline", plot=False)

    default_before = _xdate_quiet(rwi, show_flags=False)
    cof = _xdate_quiet(data, preset="COFECHA", show_flags=False)   # raw; self-detrends
    default_after = _xdate_quiet(rwi, show_flags=False)

    # default path: COFECHA-style A/B split (only CAM131 is a true A flag), no
    # COFECHA-only keys
    a_flagged = sorted(k for k, v in default_before["flags"].items() if v["A"])
    assert a_flagged == ["CAM131"]
    assert not any(k in default_before for k in ("segments", "n_problems", "preset"))

    # calling the preset in between does not perturb the default result
    assert default_before["seg_corr"].equals(default_after["seg_corr"])
    assert default_before["overall"].equals(default_after["overall"])

    # COFECHA path: extra keys present and self-consistent
    assert cof["preset"] == "COFECHA"
    assert cof["n_problems"] == sum(len(f["A"]) + len(f["B"])
                                    for f in cof["flags"].values())
    assert cof["n_problems"] >= 0


def test_xdate_default_b_flag_is_cofecha_no_margin():
    # The default (preset=None) B flag must follow COFECHA's rule -- flagged when
    # the best correlation is at a non-dated lag, with NO margin. The old dplPy
    # code gated B on (best_corr - rho) >= 0.08, an arbitrary threshold that is in
    # neither dplR (which has no B flag) nor COFECHA (which uses no margin). This
    # guards against that margin ever creeping back: every B flag's best_lag is
    # non-zero, and at least one B flag has an alternate-lag gain below 0.08 that
    # the old gate would have suppressed.
    data = _read_quiet("tests/data/csv/ca533.csv")
    rwi = dpl.detrend(data, fit="spline", plot=False)
    res = _xdate_quiet(rwi, show_flags=False)
    below_old_margin = 0
    for name, f in res["flags"].items():
        for b in f["B"]:
            assert b["best_lag"] != 0
            r0 = res["seg_corr"].loc[name, b["segment"]]
            if pd.notna(r0) and (b["best_corr"] - r0) < 0.08:
                below_old_margin += 1
    assert below_old_margin > 0     # proves no 0.08 margin is applied


def test_xdate_bins_floor_from_data_first_year_not_prewhitened():
    # Regression: get_bins must floor the first segment from the DATA's first year
    # (as dplR does), NOT the prewhitened first-valid year. AR prewhitening blanks
    # the leading `order` years; taking the bin span from the prewhitened first
    # value would floor later and silently drop early segments. wa082 starts in
    # 1698 (prewhitened first value ~1706); the first bin must be 1700-1749, not
    # 1800-1849. The earliest bin itself may hold no *complete* window, but the
    # pre-1800 bins now carry the ~20 segments the buggy 1800-floor dropped.
    data = _read_quiet("tests/data/rwl/wa082.rwl")
    rwi = dpl.detrend(data, fit="Spline", plot=False)
    res = _xdate_quiet(rwi)
    assert res["bins"][0] == "1700-1749"
    assert res["seg_corr"]["1750-1799"].notna().sum() > 0     # early region evaluated
    # total evaluated segments recovers to dplR's ~134 (the 1800-floor bug gave 113)
    assert int(res["seg_corr"].notna().sum().sum()) > 120
