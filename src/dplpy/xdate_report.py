__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2026  OpenDendro
   GNU GPLv3
"""
__license__ = "GNU GPLv3"

# Title: xdate_report.py
# Project: OpenDendro dplPy
# Description: Batch crossdating QA/QC report, formatted in the style of COFECHA's
#              output (the .txt files the ITRDB posts). For each .rwl file it reads
#              (salvage mode), detrends, cross-dates with dpl.xdate(), collates the
#              per-series statistics, and writes a COFECHA-style text report
#              (metadata header + PART 5 segment correlations + PART 7 descriptive
#              statistics). Meant for batch QA of new ITRDB submissions -- COFECHA
#              itself is not built to run over many files.
#
# NOTE: By default this is a dplR-faithful report styled after COFECHA. Several
# columns match COFECHA closely (e.g. per-series correlation with the master, mean
# sensitivity), but the default detrending / prewhitening / flagging follow dplR,
# so figures such as the overall intercorrelation and problem-segment counts will
# not be bit-identical to the COFECHA Fortran. Pass preset="COFECHA" to emulate
# the COFECHA program (Burg prewhitening, spline variance stabilization, its
# segment anchoring / critical value, omit-absent-rings, length-weighted summary),
# which reproduces COFECHA's headline numbers closely (see dpl.xdate preset).
#
# example usage:
# >>> import dplpy as dpl
# >>> dpl.xdate_report("AK200.rwl", out_dir="qc")                 # dplR-native
# >>> dpl.xdate_report("AK200.rwl", out_dir="qc", preset="COFECHA")
# >>> dpl.xdate_report(files, out_dir="qc", preset="COFECHA")     # batch

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd

from .readers import readers
from .detrend import detrend
from .xdate import xdate, _cofecha_crit
from .sensitivity import sens1
from .autoreg import autoreg


def _lag1(x):
    """Lag-1 autocorrelation of a 1-D array (NaNs dropped)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 3 or np.std(a) == 0:
        return np.nan
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def _ar_order(series, method="yw", first_aic_min=False, max_lag=10):
    """Order of the AR model selected for a series (number of lag terms). Defaults
    mirror xdate's dplR-faithful prewhitening (Yule-Walker); the COFECHA preset
    passes method="burg", first_aic_min=True to match its Burg prewhitening."""
    try:
        params = autoreg(series.dropna(), max_lag=max_lag, method=method,
                         first_aic_min=first_aic_min)
        return max(0, len(params) - 1)          # minus the constant
    except Exception:
        return 0


def _series_row(name, seq, raw, rwi_col, seg_corr_row, overall_rho, flags,
                ar_kw=None):
    """One PART-7 descriptive-statistics record for a series."""
    r = raw.dropna()
    fyr, lyr = int(r.index.min()), int(r.index.max())
    n_seg = int(seg_corr_row.notna().sum())
    n_flag = len(flags.get("A", [])) + len(flags.get("B", []))
    rwi = rwi_col.dropna()
    return {
        "seq": seq, "series": name, "first": fyr, "last": lyr,
        "nyears": int(r.size), "nseg": n_seg, "nflag": n_flag,
        "corr": float(overall_rho) if pd.notna(overall_rho) else np.nan,
        # unfiltered (raw ring widths)
        "mean": float(r.mean()), "max": float(r.max()),
        "std": float(r.std(ddof=1)) if r.size > 1 else np.nan,
        "ac": _lag1(r.to_numpy()),
        # filtered (detrended index / sensitivity)
        "sens": float(sens1(r)),
        "fmax": float(rwi.max()) if rwi.size else np.nan,
        "fstd": float(rwi.std(ddof=1)) if rwi.size > 1 else np.nan,
        "fac": _lag1(rwi.to_numpy()),
        "ar": _ar_order(raw, **(ar_kw or {})),
    }


def _process_one(path, fit, corr, slide_period, bin_floor, p_val,
                 preset=None, spline_period=None):
    """Read + detrend + cross-date one file; return an assembled report dict."""
    is_cofecha = preset is not None and str(preset).strip().lower() == "cofecha"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # join=False keeps same-ID disjoint blocks as SEPARATE series, matching
        # COFECHA's per-segment convention (so CM82S / CM86N count as two each).
        rw = readers(path, strict=False, join=False)
    if rw is None or rw.shape[1] == 0:
        raise ValueError("no usable series")
    meta = rw.attrs.get("dplpy_metadata", {}) or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_cofecha:
            # COFECHA emulation: a 32-yr spline detrend feeding xdate's COFECHA
            # preset, with absent rings (zeros in the raw frame) omitted.
            rwi = detrend(rw, fit="Spline", period=(spline_period or 32), plot=False)
            res = xdate(rwi, preset="COFECHA", absent=rw, slide_period=slide_period,
                        show_flags=False, make_plot=False)
        else:
            rwi = detrend(rw, fit=fit, plot=False)
            res = xdate(rwi, corr=corr, slide_period=slide_period, bin_floor=bin_floor,
                        p_val=p_val, show_flags=False, make_plot=False)

    seg_corr, overall, flags = res["seg_corr"], res["overall"], res["flags"]
    bins = res["bins"]
    # The COFECHA "Filtered" columns describe the PREWHITENED series (AR-removed, ~0
    # autocorrelation), which xdate returns as res["rwi"] -- not the merely detrended
    # index (which keeps its autocorrelation).
    pw = res.get("rwi")
    # AR order reported in PART 7 mirrors the prewhitening actually used: Burg /
    # first-local-AIC-min for the COFECHA preset, Yule-Walker for the default.
    ar_kw = ({"method": "burg", "first_aic_min": True, "max_lag": 10}
             if is_cofecha else {"method": "yw", "max_lag": 10})
    rows = []
    for seq, name in enumerate(seg_corr.index, start=1):
        fseries = pw[name] if (pw is not None and name in pw.columns) else pd.Series(dtype=float)
        rows.append(_series_row(
            name, seq, rw[name], fseries,
            seg_corr.loc[name], overall.loc[name, "rho"], flags.get(name, {}),
            ar_kw=ar_kw))
    return {
        "path": path, "meta": meta, "rows": rows, "bins": bins,
        "seg_corr": seg_corr, "flags": flags,
        "avg_seg_corr": res.get("avg_seg_corr"),
        "first_year": int(rw.index.min()), "last_year": int(rw.index.max()),
        "n_series": rw.shape[1], "slide_period": slide_period,
        "preset": "COFECHA" if is_cofecha else None,
    }


# --- text formatting (COFECHA-style) ----------------------------------------

def _z(v, dec):
    """Format a float with ``dec`` decimals, dropping the leading zero for values
    below 1 in magnitude (COFECHA/dplR style: .564, -.032). NaN -> empty."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = ("%." + str(dec) + "f") % v
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def _cell(v, flag=""):
    """A 5-column PART-5 correlation cell: a space, then the 2-decimal value (no
    leading zero) with an optional A/B flag, left-justified in 4 (e.g. ' .75 ',
    ' .29B', ' -.03'). Blank when missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "     "
    return " %-4s" % (_z(v, 2) + (flag or ""))


def _summary_stats(rows, weighted=False):
    corr = np.array([r["corr"] for r in rows], float)
    sens = np.array([r["sens"] for r in rows], float)
    std = np.array([r["std"] for r in rows], float)
    ac = np.array([r["ac"] for r in rows], float)
    wts = np.array([r["nyears"] for r in rows], float)
    n_flag = sum(r["nflag"] for r in rows)
    n_seg = sum(r["nseg"] for r in rows)

    def _mean(v):
        # COFECHA weights every summary statistic by series length (ring count),
        # e.g. ZSEN = sum(SEN*N) / sum(N); the dplR-native report uses a plain mean.
        ok = ~np.isnan(v)
        if not ok.any():
            return np.nan
        if weighted:
            return float(np.sum(v[ok] * wts[ok]) / np.sum(wts[ok]))
        return float(np.mean(v[ok]))

    return {
        "intercorr": _mean(corr),
        "sens": _mean(sens),
        "std": _mean(std),
        "ac": _mean(ac),
        "n_flag": n_flag, "n_seg": n_seg,
        "pct_flag": (100.0 * n_flag / n_seg) if n_seg else 0.0,
    }


def _format_header(rep):
    m = rep["meta"]
    weighted = bool(rep.get("preset"))
    s = _summary_stats(rep["rows"], weighted=weighted)
    L = []

    def fld(label, val):
        L.append("      %-23s: %s" % (label, val))

    engine = "dplPy"
    try:
        import dplpy as _d
        v = getattr(_d, "__version__", "")
        if v:
            engine += " " + v
    except Exception:
        pass
    mode = ' preset="COFECHA"' if weighted else ""
    L.append("")
    L.append("COFECHA-style output, reproduced with %s%s on %s"
             % (engine, mode, date.today().strftime("%d%b%y")))
    L.append("")
    L.append("")

    fld("Measurement file name", os.path.basename(rep["path"]))
    # Optional ITRDB header fields -- shown only when the .rwl carries them.
    if m.get("site_name"):
        fld("Site name", m["site_name"])
    if m.get("country_region"):
        fld("Site location", m["country_region"])
    sp = (str(m.get("species_code", "") or "").strip() + " "
          + str(m.get("species_name", "") or "").strip()).strip()
    if sp:
        fld("Species information", sp)
    if m.get("investigators"):
        fld("Principal investigators", m["investigators"])
    if m.get("latitude") is not None:
        fld("Latitude", "%.4f" % m["latitude"])
        fld("Longitude", "%.4f" % m["longitude"])
    if m.get("elevation_m") is not None:
        fld("Elevation", str(m["elevation_m"]) + "M")
    fld("Beginning year", rep["first_year"])
    fld("Ending year", rep["last_year"])
    L.append("")
    fld("Series intercorrelation", _z(s["intercorr"], 3))
    fld("Avg mean sensitivity", _z(s["sens"], 3))
    fld("Avg standard deviation", _z(s["std"], 3))
    fld("Avg autocorrelation", _z(s["ac"], 3))
    fld("Number dated series", rep["n_series"])
    fld("Segment length tested", rep["slide_period"])
    L.append("")
    fld("Number problem segments", s["n_flag"])
    fld("Pct problem segments", "%.2f" % s["pct_flag"])
    L.append("")
    return "\n".join(L)


_P5_PREFIX = 25          # columns start here; keeps the 131-col rule comfortable


def _format_part5(rep, per_block=20):
    seg_corr, flags, bins = rep["seg_corr"], rep["flags"], rep["bins"]
    starts = [int(b.split("-")[0]) for b in bins]
    ends = [int(b.split("-")[1]) for b in bins]
    sp = rep["slide_period"]
    avg = rep.get("avg_seg_corr")

    if rep.get("preset"):
        # COFECHA's critical r depends on the segment (window) length -- it is the
        # 99% one-tailed t value on (n-2) df converted to r. For the standard
        # 50-year window this is .3281; a different slide_period gives a different
        # threshold, so derive it from sp rather than hardcoding.
        crit = _cofecha_crit(sp)
        crit_str = ("%.4f" % crit).lstrip("0")
        crit_phrase = "under %s but highest as dated" % crit_str
        pcrit_line = ("        critical value %s for the %d-year window, "
                      "from pcrit = 0.01, one tailed" % (crit_str, sp))
    else:
        crit_phrase = "not significant but highest as dated"
        pcrit_line = "        critical value from pcrit = 0.05, one tailed"

    out = [" PART 5:  CORRELATION OF SERIES BY SEGMENTS: ",
           "-" * 131,
           "Correlations of  %d-year dated segments, lagged  %d years" % (sp, sp // 2),
           "Flags:  A = correlation %s;  B = correlation higher at other than dated position"
           % crit_phrase,
           pcrit_line, ""]

    seq_of = {name: i + 1 for i, name in enumerate(seg_corr.index)}
    for c0 in range(0, len(bins), per_block):
        cols = list(range(c0, min(c0 + per_block, len(bins))))
        out.append(("%-*s" % (_P5_PREFIX, " Seq Series  Time_span"))
                   + "".join("%5d" % starts[c] for c in cols))
        out.append((" " * _P5_PREFIX) + "".join("%5d" % ends[c] for c in cols))
        out.append(("%-*s" % (_P5_PREFIX, " --- -------- ---------"))
                   + "".join(" ----" for _ in cols))
        for name in seg_corr.index:
            row = seg_corr.loc[name]
            if all(pd.isna(row.iloc[c]) for c in cols):
                continue                                    # nothing in this block
            # A and B flags are both dicts carrying the bin label under "segment"
            fa = set(d["segment"] for d in flags.get(name, {}).get("A", []))
            fb = set(d["segment"] for d in flags.get(name, {}).get("B", []))
            r = rep["_rowmap"][name]
            cells = []
            for c in cols:
                b = bins[c]
                fl = "A" if b in fa else ("B" if b in fb else "")
                cells.append(_cell(row.iloc[c], fl))
            out.append(" %3d %-8s %4d %4d %s"
                       % (seq_of[name], name[:8], r["first"], r["last"], "".join(cells)))
        # mean correlation across series for each segment in this block
        if avg is not None:
            av_cells = []
            for c in cols:
                v = avg.iloc[c] if hasattr(avg, "iloc") else avg[c]
                av_cells.append("  NaN" if pd.isna(v) else _cell(v))
            out.append(("%-*s" % (_P5_PREFIX, " Av segment correlation"))
                       + "".join(av_cells))
        out.append("")
    return "\n".join(out)


_P7_HEAD = [
    "                                                Corr   "
    "//-------- Unfiltered --------\\\\  //---- Filtered -----\\\\",
    "                           No.    No.    No.    with   "
    "Mean   Max     Std   Auto   Mean   Max     Std   Auto  AR",
    " Seq Series   Interval   Years  Segmt  Flags   Master  "
    "msmt   msmt    dev   corr   sens  value    dev   corr  ()",
    " --- -------- ---------  -----  -----  -----   ------ "
    "-----  -----  -----  -----  -----  -----  -----  -----  --",
]


def _p7_row(seq, name, first, last, nyears, nseg, nflag,
            corr, mean, mx, std, ac, sens, fmax, fstd, fac, ar):
    """One PART-7 data line, aligned under _P7_HEAD (widths match COFECHA/dplR)."""
    return (" %3d %-8s %4d %4d  %5s  %5s  %5s   %6s"
            " %5s  %5s  %5s  %5s  %5s  %5s  %5s  %5s  %2s"
            % (seq, name[:8], first, last,
               _blank(nyears, "%d"), _blank(nseg, "%d"), _blank(nflag, "%d"),
               _z(corr, 3), _z(mean, 2), _z(mx, 2), _z(std, 3), _z(ac, 3),
               _z(sens, 3), _z(fmax, 2), _z(fstd, 3), _z(fac, 3),
               _blank(ar, "%d")))


def _blank(v, fmt):
    """Format an int with ``fmt``, or blank if None/NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return fmt % v


def _format_part7(rep):
    out = [" PART 7:  DESCRIPTIVE STATISTICS: ", "-" * 131, ""]
    out.extend(_P7_HEAD)
    for r in rep["rows"]:
        out.append(_p7_row(
            r["seq"], r["series"], r["first"], r["last"], r["nyears"],
            r["nseg"], r["nflag"], r["corr"], r["mean"], r["max"], r["std"],
            r["ac"], r["sens"], r["fmax"], r["fstd"], r["fac"], r["ar"]))
    s = _summary_stats(rep["rows"], weighted=bool(rep.get("preset")))
    n_years = sum(x["nyears"] for x in rep["rows"])
    n_seg = sum(x["nseg"] for x in rep["rows"])

    # column means for the total row: length-weighted in COFECHA mode (as COFECHA
    # reports them), a plain mean otherwise.
    def _wm(field, dec):
        v = np.array([x[field] for x in rep["rows"]], float)
        w = np.array([x["nyears"] for x in rep["rows"]], float)
        ok = ~np.isnan(v)
        if not ok.any():
            return ""
        m = (np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if rep.get("preset") \
            else float(np.mean(v[ok]))
        return _z(m, dec)

    out.append(_P7_HEAD[-1])
    # label fills the seq+series+interval span; the numeric columns then line up
    # under the data rows (years starts at column 25).
    out.append(("%-25s" % " Total or mean:")
               + "%5s  %5s  %5s   %6s %5s  %5s  %5s  %5s  %5s  %5s  %5s  %5s"
               % (_blank(n_years, "%d"), _blank(n_seg, "%d"), s["n_flag"],
                  _z(s["intercorr"], 3), _wm("mean", 2), _wm("max", 2),
                  _z(s["std"], 3), _z(s["ac"], 3), _z(s["sens"], 3),
                  _wm("fmax", 2), _wm("fstd", 3), _wm("fac", 3)))
    return "\n".join(out)


def _format_report(rep):
    rep["_rowmap"] = {r["series"]: r for r in rep["rows"]}
    return (_format_header(rep) + "\n\n" + _format_part5(rep)
            + "\n" + _format_part7(rep) + "\n")


def xdate_report(files, out_dir=".", fit="Spline", corr="spearman",
                 slide_period=50, bin_floor=100, p_val=0.05,
                 output="both", verbose=True, preset=None, spline_period=None):
    """Generate a COFECHA-style crossdating QA report for one or more .rwl files.

    For each file: read (salvage mode), detrend, cross-date with ``dpl.xdate``,
    collate per-series statistics, and print the report and/or save it as a
    ``<name>.txt`` in ``out_dir`` (see ``output``). Built for batch QA of ITRDB
    submissions.

    By default this is a dplR-faithful report styled after COFECHA (some columns
    match COFECHA closely, others differ by method). Pass ``preset="COFECHA"`` to
    emulate the COFECHA program instead: each file is detrended with a 32-year
    spline and cross-dated with ``dpl.xdate(preset="COFECHA")`` (Burg prewhitening,
    spline variance stabilization, an arithmetic z-scored master, Pearson
    correlation, COFECHA segment anchoring and critical value, and "omit absent
    rings" fed automatically from the raw frame), and the summary statistics are
    length-weighted by ring count as COFECHA reports them.

    Parameters
    ----------
    files : str or list of str
        A .rwl path, or a list of them.
    out_dir : str, default "."
        Directory for the ``.txt`` reports (created if needed).
    fit : str, default "Spline"
        detrending curve passed to ``dpl.detrend`` (ignored when
        ``preset="COFECHA"``, which always uses a spline).
    corr, slide_period, bin_floor, p_val
        passed through to ``dpl.xdate`` (``corr``, ``bin_floor`` and ``p_val`` are
        ignored when ``preset="COFECHA"``; ``slide_period`` still applies).
    output : {"both", "screen", "file"}, default "both"
        where the full report goes: ``"screen"`` prints it to the notebook/console
        for immediate reading, ``"file"`` writes a ``<name>.txt`` in ``out_dir``,
        ``"both"`` does both. The report text is always available in the returned
        dict regardless of this setting.
    verbose : bool, default True
        print a one-line per-file progress note and a final tally (separate from
        ``output``, which controls the full report text).
    preset : str or None, default None
        set to ``"COFECHA"`` to emulate the COFECHA program (see above).
    spline_period : int or None, default None
        spline stiffness (years) for the ``preset="COFECHA"`` detrend; defaults
        to COFECHA's 32.

    Returns
    -------
    dict
        ``{path: {"text": str, "report": dict}}`` for files read successfully, or
        ``{path: {"error": str}}`` for files that failed.
    """
    output = str(output).strip().lower()
    if output not in ("screen", "file", "both"):
        raise ValueError("output must be 'screen', 'file', or 'both', got %r." % output)
    to_screen = output in ("screen", "both")
    to_file = output in ("file", "both")

    if isinstance(files, str):
        files = [files]
    if to_file:
        os.makedirs(out_dir, exist_ok=True)
    results = {}
    ok = 0
    for i, path in enumerate(files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            rep = _process_one(path, fit, corr, slide_period, bin_floor, p_val,
                               preset=preset, spline_period=spline_period)
            text = _format_report(rep)
            if to_file:
                with open(os.path.join(out_dir, base + ".txt"), "w") as fh:
                    fh.write(text)
            if to_screen:
                print(text)
            results[path] = {"text": text, "report": rep}
            ok += 1
            if verbose:
                print("  [%d/%d] %s: %d series, %d problem segments"
                      % (i, len(files), base, rep["n_series"],
                         _summary_stats(rep["rows"])["n_flag"]))
        except Exception as e:
            results[path] = {"error": str(e)}
            if verbose:
                print("  [%d/%d] %s: FAILED -- %s" % (i, len(files), base, str(e)[:80]))
    if verbose:
        print("%d of %d file(s) reported OK%s"
              % (ok, len(files), (" -> " + out_dir) if to_file else ""))
    return results
