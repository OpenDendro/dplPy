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
# NOTE: This is a dplPy-NATIVE report. Several columns match COFECHA closely (e.g.
# per-series correlation with the master, mean sensitivity), but dplPy's
# detrending / prewhitening / flagging differ from the COFECHA Fortran, so figures
# such as the overall intercorrelation and problem-segment counts will not be
# bit-identical. A future preset="COFECHA" could tighten the match.
#
# example usage:
# >>> import dplpy as dpl
# >>> dpl.xdate_report("AK200.rwl", out_dir="qc")            # one file
# >>> dpl.xdate_report(files, out_dir="qc")                  # a list of paths

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd

from .readers import readers
from .detrend import detrend
from .xdate import xdate
from .sensitivity import sens1
from .autoreg import autoreg


def _lag1(x):
    """Lag-1 autocorrelation of a 1-D array (NaNs dropped)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 3 or np.std(a) == 0:
        return np.nan
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def _ar_order(series):
    """Order of the AR model dpl.autoreg selects (number of lag terms)."""
    try:
        params = autoreg(series.dropna())
        return max(0, len(params) - 1)          # minus the constant
    except Exception:
        return 0


def _series_row(name, seq, raw, rwi_col, seg_corr_row, overall_rho, flags):
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
        "ar": _ar_order(raw),
    }


def _process_one(path, fit, corr, slide_period, bin_floor, p_val):
    """Read + detrend + cross-date one file; return an assembled report dict."""
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
        rwi = detrend(rw, fit=fit, plot=False)
        res = xdate(rwi, corr=corr, slide_period=slide_period, bin_floor=bin_floor,
                    p_val=p_val, show_flags=False, make_plot=False)

    seg_corr, overall, flags = res["seg_corr"], res["overall"], res["flags"]
    bins = res["bins"]
    # The COFECHA "Filtered" columns describe the PREWHITENED series (AR-removed, ~0
    # autocorrelation), which xdate returns as res["rwi"] -- not the merely detrended
    # index (which keeps its autocorrelation).
    pw = res.get("rwi")
    rows = []
    for seq, name in enumerate(seg_corr.index, start=1):
        fseries = pw[name] if (pw is not None and name in pw.columns) else pd.Series(dtype=float)
        rows.append(_series_row(
            name, seq, rw[name], fseries,
            seg_corr.loc[name], overall.loc[name, "rho"], flags.get(name, {})))
    return {
        "path": path, "meta": meta, "rows": rows, "bins": bins,
        "seg_corr": seg_corr, "flags": flags,
        "avg_seg_corr": res.get("avg_seg_corr"),
        "first_year": int(rw.index.min()), "last_year": int(rw.index.max()),
        "n_series": rw.shape[1], "slide_period": slide_period,
    }


# --- text formatting (COFECHA-style) ----------------------------------------

def _fmt_corr(v, flag=""):
    """A correlation cell in COFECHA style: leading zero dropped, optional A/B
    flag, 5 columns wide (blank when missing)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "     "
    s = ("%.2f" % v)
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return "%4s%s" % (s, (flag or " "))


def _summary_stats(rows):
    corr = np.array([r["corr"] for r in rows], float)
    sens = np.array([r["sens"] for r in rows], float)
    std = np.array([r["std"] for r in rows], float)
    ac = np.array([r["ac"] for r in rows], float)
    n_flag = sum(r["nflag"] for r in rows)
    n_seg = sum(r["nseg"] for r in rows)
    return {
        "intercorr": np.nanmean(corr) if corr.size else np.nan,
        "sens": np.nanmean(sens) if sens.size else np.nan,
        "std": np.nanmean(std) if std.size else np.nan,
        "ac": np.nanmean(ac) if ac.size else np.nan,
        "n_flag": n_flag, "n_seg": n_seg,
        "pct_flag": (100.0 * n_flag / n_seg) if n_seg else 0.0,
    }


def _format_header(rep):
    m = rep["meta"]
    s = _summary_stats(rep["rows"])
    L = []
    L.append(" " + str(m.get("site_name", "")) + "   - " + str(m.get("site_id", "")))
    L.append("Additional Site Information")
    L.append(" " + str(m.get("investigators", "")))
    L.append("")
    L.append("      Report generated by   : dplPy (dpl.xdate_report)")
    L.append("      Measurement file name  : " + os.path.basename(rep["path"]))
    L.append("      Date checked           : " + date.today().strftime("%d%b%y").upper())
    L.append("      Beginning year         : " + str(rep["first_year"]))
    L.append("      Ending year            : " + str(rep["last_year"]))
    L.append("      Principal investigators: " + str(m.get("investigators", "")))
    L.append("      Site name              : " + str(m.get("site_name", "")))
    L.append("      Site location          : " + str(m.get("country_region", "")))
    L.append("      Species information    : " + str(m.get("species_code", ""))
             + " " + str(m.get("species_name", "")))
    if m.get("latitude") is not None:
        L.append("      Latitude               : %.4f" % m["latitude"])
        L.append("      Longitude              : %.4f" % m["longitude"])
    if m.get("elevation_m") is not None:
        L.append("      Elevation              : " + str(m["elevation_m"]) + "M")
    L.append("")
    L.append("      Series intercorrelation: %6.3f" % s["intercorr"])
    L.append("      Avg mean sensitivity   : %6.3f" % s["sens"])
    L.append("      Avg standard deviation : %6.3f" % s["std"])
    L.append("      Avg autocorrelation    : %6.3f" % s["ac"])
    L.append("      Number dated series    : " + str(rep["n_series"]))
    L.append("      Segment length tested  : " + str(rep["slide_period"]))
    L.append("")
    L.append("      Number problem segments: " + str(s["n_flag"]))
    L.append("      Pct problem segments   : %5.2f" % s["pct_flag"])
    L.append("")
    return "\n".join(L)


def _format_part5(rep, per_block=20):
    seg_corr, flags, bins = rep["seg_corr"], rep["flags"], rep["bins"]
    starts = [int(b.split("-")[0]) for b in bins]
    ends = [int(b.split("-")[1]) for b in bins]
    out = ["PART 5:  CORRELATION OF SERIES BY SEGMENTS",
           "-" * 100,
           " Correlations of %2d-year dated segments, lagged %2d years"
           % (rep["slide_period"], rep["slide_period"] // 2),
           " Flags:  A = correlation not significant but highest as dated;"
           "  B = correlation higher at another position", ""]
    seq_of = {name: i + 1 for i, name in enumerate(seg_corr.index)}
    for c0 in range(0, len(bins), per_block):
        cols = list(range(c0, min(c0 + per_block, len(bins))))
        out.append(" Seq Series   Time_span  "
                   + "".join("%5d" % starts[c] for c in cols))
        out.append("                          "
                   + "".join("%5d" % ends[c] for c in cols))
        out.append(" --- --------  --------- " + "-----" * len(cols))
        for name in seg_corr.index:
            row = seg_corr.loc[name]
            if all(pd.isna(row.iloc[c]) for c in cols):
                continue                                    # nothing in this block
            # A flags are bin labels; B flags are dicts carrying the bin under "segment"
            fa = set(flags.get(name, {}).get("A", []))
            fb = set(d["segment"] for d in flags.get(name, {}).get("B", []))
            r = rep["_rowmap"][name]
            cells = []
            for c in cols:
                b = bins[c]
                fl = "A" if b in fa else ("B" if b in fb else "")
                cells.append(_fmt_corr(row.iloc[c], fl))
            out.append("%4d %-8s %5d %4d  %s"
                       % (seq_of[name], name[:8], r["first"], r["last"], "".join(cells)))
        out.append("")
    return "\n".join(out)


def _format_part7(rep):
    out = ["PART 7:  DESCRIPTIVE STATISTICS",
           "-" * 100, "",
           "                                             Corr   "
           "//------- Unfiltered -------\\\\  //---- Filtered ----\\\\",
           "                          No.   No.   No.   with   "
           "Mean   Max    Std   Auto   Mean   Max    Std   Auto  AR",
           " Seq Series   Interval  Years Segmt Flags  Master  "
           "msmt   msmt    dev   corr   sens  value    dev   corr  ()",
           " --- --------  --------- ----- ----- -----  ------ "
           "-----  -----  -----  -----  -----  -----  -----  -----  --"]
    for r in rep["rows"]:
        out.append(
            "%4d %-8s %5d %4d  %5d %5d %5d   %s   %4.2f  %5.2f  %s  %s  %s  %5.2f  %s  %s %3d"
            % (r["seq"], r["series"][:8], r["first"], r["last"], r["nyears"],
               r["nseg"], r["nflag"], _num3(r["corr"]), r["mean"], r["max"],
               _num3(r["std"]), _num3(r["ac"]), _num3(r["sens"]), r["fmax"],
               _num3(r["fstd"]), _num3(r["fac"]), r["ar"]))
    s = _summary_stats(rep["rows"])
    out.append(" --- --------  --------- ----- ----- -----  ------ "
               "-----  -----  -----  -----  -----  -----  -----  -----  --")
    out.append(" Total or mean:          %6d %5s %5d   %s"
               % (sum(x["nyears"] for x in rep["rows"]), "",
                  s["n_flag"], _num3(s["intercorr"])))
    return "\n".join(out)


def _num3(v):
    """A 3-decimal figure with the leading zero dropped, 6 columns (COFECHA style)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "     ."
    s = "%.3f" % v
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return "%6s" % s


def _format_report(rep):
    rep["_rowmap"] = {r["series"]: r for r in rep["rows"]}
    return (_format_header(rep) + "\n\n" + _format_part5(rep)
            + "\n" + _format_part7(rep) + "\n")


def xdate_report(files, out_dir=".", fit="Spline", corr="spearman",
                 slide_period=50, bin_floor=100, p_val=0.05,
                 write=True, verbose=True):
    """Generate a COFECHA-style crossdating QA report for one or more .rwl files.

    For each file: read (salvage mode), detrend, cross-date with ``dpl.xdate``,
    collate per-series statistics, and (if ``write``) save a ``<name>.txt`` report
    in ``out_dir``. Built for batch QA of ITRDB submissions.

    This is a dplPy-native report styled after COFECHA; some columns match COFECHA
    closely while others differ by method (see the module note).

    Parameters
    ----------
    files : str or list of str
        A .rwl path, or a list of them.
    out_dir : str, default "."
        Directory for the ``.txt`` reports (created if needed).
    fit : str, default "Spline"
        detrending curve passed to ``dpl.detrend``.
    corr, slide_period, bin_floor, p_val
        passed through to ``dpl.xdate``.
    write : bool, default True
        write the ``.txt`` files; if False, only the text is returned.
    verbose : bool, default True
        print progress and a final tally.

    Returns
    -------
    dict
        ``{path: {"text": str, "report": dict}}`` for files read successfully, or
        ``{path: {"error": str}}`` for files that failed.
    """
    if isinstance(files, str):
        files = [files]
    if write:
        os.makedirs(out_dir, exist_ok=True)
    results = {}
    ok = 0
    for i, path in enumerate(files, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            rep = _process_one(path, fit, corr, slide_period, bin_floor, p_val)
            text = _format_report(rep)
            if write:
                with open(os.path.join(out_dir, base + ".txt"), "w") as fh:
                    fh.write(text)
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
              % (ok, len(files), (" -> " + out_dir) if write else ""))
    return results
