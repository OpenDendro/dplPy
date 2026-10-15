__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2026  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
__license__ = "GNU GPLv3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Title: check.py
# Project: OpenDendro dplPy
# Description: dpl.check() -- integrity checks for a ring-width collection,
#              reported as data. Modeled on dplR 1.8.0's rwl.check() (A.G. Bunn):
#              report() answers "what does this file look like", check() answers
#              "what is wrong with it", as a table of findings so that a report
#              over one file and a sweep over a whole archive are the same code.
#              Each finding carries a stable check id and a severity. Every check
#              runs inside a guard and a check that fails becomes a finding rather
#              than an error. Built on dplPy's own reader, spline, and crossdating
#              pieces, so it keeps dplPy's read tolerance rather than dplR's.

import os
import glob
import re
import warnings
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ._validate import (_require_dataframe, _require_year_index, _alias_kwarg,
                        _reject_kwargs)
from .smoothingspline import _caps
from .xdate import _lag_table

# Reader diagnostics that check() re-reports as provenance/structure findings
# (RWL_SALVAGED, RWL_ID_RENAMED, RWL_ENCODING_ASSUMED, RWL_INTERNAL_NA, ...).
# Only these messages are silenced while reading; anything else still surfaces
# (audit G3: no blanket warnings.simplefilter("ignore")).
_READER_NOISE = re.compile(
    r".*(?:Salvaged|duplicate ID|case-variant|letter case|latin1|stop marker|interior gap"
    r"|beyond the last data column|not recognized; inferred|could not be parsed"
    r"|misaligned row|non-integer value|gap sentinel|anomalous negative"
    r"|missing line break|fixed-width parse|formatting problems"
    r"|identical duplicate|header=True but no header|returning None)", re.I)

_FINDING_COLS = ["file", "check", "severity", "group", "series",
                 "year_from", "year_to", "n", "value", "message"]

# One row per check. Keep this table and the check code in step. Check ids match
# dplR 1.8.0 (RWL_ prefix) so a maintainer using both tools reads the same ids.
_CATALOGUE = [
    ("RWL_NO_SERIES", "structure", "error", "collection contains no series"),
    ("RWL_ALL_NA_YEAR", "structure", "warning", "years where every series is NA"),
    ("RWL_LOW_DEPTH", "structure", "note", "sustained years below min_depth series"),
    ("RWL_FUTURE_YEAR", "structure", "error", "years later than max_year"),
    ("RWL_INTERNAL_NA", "series", "warning", "NA inside a series' measured span"),
    ("RWL_EMPTY_SERIES", "series", "warning", "series has no measurements"),
    ("RWL_DUP_SERIES", "series", "error", "two or more series are identical"),
    ("RWL_ZERO_VARIANCE", "series", "error", "series is constant"),
    ("RWL_SHORT_SERIES", "series", "note", "series shorter than min_length"),
    ("RWL_REPEATED_VALUE", "series", "warning", "same non-zero value repeated"),
    ("RWL_ID_PATTERN", "series", "note", "id departs from the dominant pattern"),
    ("RWL_SITE_CODE", "series", "warning", "id carries a different site code"),
    ("RWL_NEGATIVE", "values", "error", "negative ring width"),
    ("RWL_GRANULARITY", "values", "note", "coarse inferred precision"),
    ("RWL_IMPLAUSIBLE_MEAN", "values", "warning", "mean outside a plausible range"),
    ("RWL_SMALL_RING", "values", "note", "ring width below small_thresh"),
    ("RWL_BIG_RING", "values", "note", "ring width above big_thresh"),
    ("RWL_ZERO_RING", "zeros", "note", "ring width of zero (locally absent)"),
    ("RWL_ALL_ZERO_YEAR", "zeros", "warning", "years where every series is zero"),
    ("RWL_CONSECUTIVE_ZERO", "zeros", "note", "more than one consecutive zero"),
    ("RWL_DATING_LAG", "crossdating", "error", "best correlation at a non-zero lag"),
    ("RWL_SERIES_OUTLIER", "crossdating", "warning", "series fits the collection poorly"),
    ("RWL_WEAK_COLLECTION", "crossdating", "note", "little common signal in the collection"),
    ("RWL_UNCHECKABLE", "crossdating", "note", "series cannot be detrended, so not checked"),
    ("RWL_SHORT_OVERLAP", "crossdating", "note", "overlaps the master by too few rings"),
    ("RWL_MIXED_EOL", "file", "warning", "file mixes CRLF and LF line endings"),
    ("RWL_TAB", "file", "warning", "file contains tab characters"),
    ("RWL_NON_ASCII", "file", "warning", "file contains non-ASCII bytes"),
    ("RWL_NO_FINAL_NEWLINE", "file", "note", "file does not end with a newline"),
    ("RWL_HEADER_SPAN", "file", "warning", "header span disagrees with the data"),
    ("RWL_ENCODING_ASSUMED", "provenance", "warning", "read as latin1, no encoding declared"),
    ("RWL_MIXED_PRECISION", "provenance", "warning", "file declares more than one precision"),
    ("RWL_SERIES_DROPPED", "provenance", "error", "a series in the file is not in the object"),
    ("RWL_ID_RENAMED", "provenance", "warning", "the reader renamed or merged a series"),
    ("RWL_SALVAGED", "provenance", "warning", "the reader recovered from a defect on read"),
    ("RWL_CHECK_ERROR", "engine", "error", "a check failed to run"),
]
_SEV = {c[0]: c[2] for c in _CATALOGUE}
_GRP = {c[0]: c[1] for c in _CATALOGUE}
_SEV_ORDER = {"error": 0, "warning": 1, "note": 2}


def _f(check, message, series=None, year_from=None, year_to=None, n=None, value=None):
    return {"check": check, "series": series, "year_from": year_from,
            "year_to": year_to, "n": n, "value": value, "message": message}


def _year_runs(years):
    """Collapse a list of years into (from, to, n) runs of consecutive years."""
    y = np.array(sorted(set(int(v) for v in years)))
    if y.size == 0:
        return []
    brk = np.r_[0, np.where(np.diff(y) != 1)[0] + 1, y.size]
    return [(int(y[brk[i]]), int(y[brk[i + 1] - 1]), int(brk[i + 1] - brk[i]))
            for i in range(len(brk) - 1)]


def _span_text(a, b):
    a, b = int(a), int(b)
    return str(a) if a == b else "%d-%d" % (a, b)


def _granularity(values):
    """Largest step dividing every measurement (inferred precision, mm)."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return np.nan
    iv = np.unique(np.round(v * 1000).astype(np.int64))
    g = 0
    for x in iv:
        a, b = g, int(x)
        while b:
            a, b = b, a % b
        g = a
    return g / 1000.0


# ---- the groups: each returns a list of finding dicts, and never raises ------

def _check_structure(M, years, cols, ctl):
    out = []
    if M.shape[1] == 0:
        return [_f("RWL_NO_SERIES", "the collection holds no series")]
    depth = np.sum(~np.isnan(M), axis=1)
    for a, b, n in _year_runs(years[depth == 0]):
        out.append(_f("RWL_ALL_NA_YEAR",
                      "no series is measured in %s" % _span_text(a, b),
                      year_from=a, year_to=b, n=n))
    low = years[(depth > 0) & (depth < ctl["min_depth"])]
    for a, b, n in _year_runs(low):
        if n >= ctl["depth_run"]:
            out.append(_f("RWL_LOW_DEPTH",
                          "%s rests on fewer than %d series" % (_span_text(a, b), ctl["min_depth"]),
                          year_from=a, year_to=b, n=n))
    fut = years[years > ctl["max_year"]]
    if fut.size:
        out.append(_f("RWL_FUTURE_YEAR",
                      "%d year(s) fall after %d" % (fut.size, ctl["max_year"]),
                      year_from=int(fut.min()), year_to=int(fut.max()), n=int(fut.size)))
    return out


def _check_series(M, years, cols, ctl):
    out = []
    import hashlib
    hashes = {}
    for j, s in enumerate(cols):
        col = M[:, j]
        ok = np.where(~np.isnan(col))[0]
        if ok.size == 0:
            out.append(_f("RWL_EMPTY_SERIES", "no measurements in any year held",
                          series=s, n=0))
            continue
        # interior NA, one finding per series
        lo, hi = ok[0], ok[-1]
        interior = np.where(np.isnan(col[lo:hi + 1]))[0] + lo
        if interior.size:
            runs = _year_runs(years[interior])
            spans = ", ".join(_span_text(a, b) for a, b, _ in runs)
            out.append(_f("RWL_INTERNAL_NA",
                          "no measurement for %s inside the series; detrend(), chron(), "
                          "rcs() and the crossdating checks fail on interior gaps. Use "
                          "fill_internal() to close them deliberately" % spans,
                          series=s, year_from=int(years[interior.min()]),
                          year_to=int(years[interior.max()]), n=int(interior.size)))
        v = col[ok]
        # duplicate detection (over finite values)
        h = hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest()
        hashes.setdefault(h, []).append(s)
        # zero variance
        if v.size > 1 and np.std(v) == 0:
            out.append(_f("RWL_ZERO_VARIANCE",
                          "every ring measures %g; a constant series is not a measurement" % v[0],
                          series=s, n=int(v.size), value=float(v[0])))
        # short
        if v.size < ctl["min_length"]:
            out.append(_f("RWL_SHORT_SERIES",
                          "%d rings, fewer than %d; too short to crossdate reliably"
                          % (v.size, ctl["min_length"]), series=s, n=int(v.size), value=int(v.size)))
        # runs of one repeated non-zero value (catches sentinels like a constant 1.0)
        yv = years[ok]
        d = np.r_[0, np.where(np.diff(v) != 0)[0] + 1, v.size]
        for i in range(len(d) - 1):
            run = d[i + 1] - d[i]
            val = v[d[i]]
            if run >= ctl["run_min"] and val > 0:
                out.append(_f("RWL_REPEATED_VALUE",
                              "%g repeated in %d consecutive rings, %s; a constant stretch is a "
                              "placeholder, not a measurement" % (val, run,
                              _span_text(yv[d[i]], yv[d[i + 1] - 1])),
                              series=s, year_from=int(yv[d[i]]), year_to=int(yv[d[i + 1] - 1]),
                              n=int(run), value=float(val)))
    for h, g in hashes.items():
        if len(g) > 1:
            out.append(_f("RWL_DUP_SERIES",
                          "identical measurements in %s; one core may be archived twice"
                          % ", ".join(g), series=g[0], n=len(g)))
    # id-pattern and site-code breaks (need a majority to compare against)
    ids = list(cols)
    if len(ids) >= 5:
        import re
        shape = [re.sub("[0-9]", "9", re.sub("[A-Za-z]", "a", str(i))) for i in ids]
        tab = pd.Series(shape).value_counts()
        if len(tab) > 1 and tab.iloc[0] >= 0.6 * len(ids):
            dom = tab.index[0]
            for s, sh in zip(ids, shape):
                if sh != dom:
                    out.append(_f("RWL_ID_PATTERN",
                                  "id does not follow the dominant pattern (%s); check whether "
                                  "it is a derived or mislabeled series" % dom, series=s))
        pre = [re.sub("[^A-Za-z].*$", "", str(i)).lower() for i in ids]
        if all(pre) and len(set(pre)) > 1:
            ptab = pd.Series(pre).value_counts()
            dom = ptab.index[0]
            if ptab.iloc[0] >= 0.75 * len(ids):
                for s, p in zip(ids, pre):
                    nested = p.startswith(dom) or dom.startswith(p)
                    if p != dom and not nested:
                        out.append(_f("RWL_SITE_CODE",
                                      "site code differs from the '%s' used by most series" % dom,
                                      series=s))
    return out


def _check_values(M, years, cols, ctl):
    out = []
    v = M[~np.isnan(M)]
    if v.size == 0:
        return out
    neg = int(np.sum(M < 0))
    if neg:
        out.append(_f("RWL_NEGATIVE",
                      "%d negative measurement(s); a ring width cannot be negative. An "
                      "unrecognized missing-value code (e.g. -9, -999, or 1 used as a "
                      "sentinel) can look like this" % neg, n=neg, value=float(np.nanmin(M))))
    g = _granularity(v)
    if np.isfinite(g) and g >= 0.1:
        out.append(_f("RWL_GRANULARITY",
                      "every measurement is a multiple of %g mm; coarser than a Tucson file "
                      "normally carries and may be rounded or in the wrong units" % g, value=float(g)))
    mu = float(np.mean(v))
    if mu < ctl["plausible_mean"][0] or mu > ctl["plausible_mean"][1]:
        out.append(_f("RWL_IMPLAUSIBLE_MEAN",
                      "mean ring width is %g mm, outside %g-%g mm; the file may be in the wrong "
                      "units or carry a sentinel value" % (round(mu, 4),
                      ctl["plausible_mean"][0], ctl["plausible_mean"][1]), value=mu))
    for thr, cid, op in (("small", "RWL_SMALL_RING", np.less),
                         ("big", "RWL_BIG_RING", np.greater)):
        t = ctl[thr + "_thresh"]
        if t is None:
            continue
        m = op(M, t) & (M > 0 if thr == "small" else ~np.isnan(M))
        m[np.isnan(M)] = False
        for j in np.where(m.sum(axis=0) > 0)[0]:
            for a, b, n in _year_runs(years[m[:, j]]):
                out.append(_f(cid, "%d ring(s) %s %g" % (int(m[:, j].sum()),
                              "below" if thr == "small" else "above", t),
                              series=cols[j], year_from=a, year_to=b, n=n))
    return out


def _check_zeros(M, years, cols, ctl):
    out = []
    m = (M == 0)
    m[np.isnan(M)] = False
    if not m.any():
        return out
    out.append(_f("RWL_ZERO_RING",
                  "%d ring(s) of zero width in %d series" % (int(m.sum()), int((m.sum(axis=0) > 0).sum())),
                  n=int(m.sum())))
    depth = np.sum(~np.isnan(M), axis=1)
    allz = (depth >= ctl["min_depth"]) & (m.sum(axis=1) == depth)
    for a, b, n in _year_runs(years[allz]):
        out.append(_f("RWL_ALL_ZERO_YEAR",
                      "every measured series is zero in %s" % _span_text(a, b),
                      year_from=a, year_to=b, n=n))
    for j, s in enumerate(cols):
        ok = ~np.isnan(M[:, j])
        if not ok.any():
            continue
        z = (M[ok, j] == 0)
        yv = years[ok]
        d = np.r_[0, np.where(np.diff(z.astype(int)) != 0)[0] + 1, z.size]
        for i in range(len(d) - 1):
            run = d[i + 1] - d[i]
            if z[d[i]] and run > 1:
                out.append(_f("RWL_CONSECUTIVE_ZERO",
                              "%d consecutive absent rings, %s" % (run,
                              _span_text(yv[d[i]], yv[d[i + 1] - 1])),
                              series=s, year_from=int(yv[d[i]]), year_to=int(yv[d[i + 1] - 1]), n=int(run)))
    return out


def _high_pass(col, nyrs):
    """Ratio of a series to a caps spline over its measured span. NA on an
    interior gap (a gapped series cannot be detrended, so cannot be checked)."""
    k = np.where(np.isfinite(col))[0]
    out = np.full(len(col), np.nan)
    if k.size < nyrs:
        return out
    lo, hi = k[0], k[-1]
    span = col[lo:hi + 1]
    if np.isnan(span).any():
        return out
    try:
        cs = np.asarray(_caps(span, nyrs, 0.5))
    except Exception:
        return out
    if np.any(cs <= 0):
        return out
    r = span / cs
    r[~np.isfinite(r)] = np.nan
    out[lo:hi + 1] = r
    return out


def _ccf_best(series, master, years, lag, min_pairs):
    """Best lag of the whole-series correlation with the master over -lag..+lag,
    via `xdate._lag_table()`: the series stays fixed and the MASTER slides, so a
    NEGATIVE best lag means a MISSING ring in the series (COFECHA / dplR
    convention, the same as xdate and series_corr). Each shifted window is
    correlated (Pearson) over its pairwise-complete pairs, needing at least
    ``min_pairs``. Returns (best_lag, best_r, r_at_lag0)."""
    fin = np.flatnonzero(np.isfinite(series))
    if fin.size == 0:
        return 0, np.nan, np.nan
    lo, hi = int(years[fin[0]]), int(years[fin[-1]])
    corrs, best_lag, best_r = _lag_table(series, master, years, lo, hi, None,
                                         "pearson", lag, min_overlap=min_pairs)
    return best_lag, best_r, float(corrs[lag])


def _check_crossdating(M, years, cols, ctl):
    out = []
    ns = M.shape[1]
    if ns < 3:
        return [_f("RWL_SHORT_OVERLAP",
                   "only %d series; too few to build a master to date against" % ns)]
    # the lag scan needs a dense one-year grid (a year gap in the index would
    # otherwise shift every later ring by one position)
    yrs_d = np.arange(int(years.min()), int(years.max()) + 1)
    if yrs_d.size != years.size:
        pos = (years - yrs_d[0]).astype(int)
        Md = np.full((yrs_d.size, ns), np.nan)
        Md[pos] = M
        M, years = Md, yrs_d.astype(float)
    HP = np.column_stack([_high_pass(M[:, j], ctl["spline_period"]) for j in range(ns)])
    r0 = np.full(ns, np.nan)
    for i in range(ns):
        s = cols[i]
        others = HP[:, [j for j in range(ns) if j != i]]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice",
                                    category=RuntimeWarning)
            master = np.nanmean(others, axis=1)
        ser = HP[:, i]
        ok = np.isfinite(ser) & np.isfinite(master)
        nraw = int(np.sum(~np.isnan(M[:, i])))
        if not np.isfinite(ser).any():
            if nraw >= ctl["min_overlap"]:
                out.append(_f("RWL_UNCHECKABLE",
                              "could not be checked against the collection: the series has gaps "
                              "inside its measured span and cannot be detrended", series=s, n=nraw))
            continue
        if ok.sum() < ctl["min_overlap"]:
            if nraw >= ctl["min_length"]:
                out.append(_f("RWL_SHORT_OVERLAP",
                              "overlaps the rest of the collection by %d rings, fewer than %d"
                              % (int(ok.sum()), ctl["min_overlap"]), series=s, n=int(ok.sum())))
            continue
        lag, r, rz = _ccf_best(ser, master, years, ctl["lag"],
                               max(10, ctl["min_overlap"] // 2))
        r0[i] = rz
        if lag != 0 and np.isfinite(r) and np.isfinite(rz) and r >= ctl["r_dating"] and (r - rz) >= ctl["r_margin"]:
            cause = "most likely a missing ring" if lag < 0 else "most likely a false ring or a ring measured twice"
            out.append(_f("RWL_DATING_LAG",
                          "correlates best with the master at lag %d (r = %.3f against r = %.3f as "
                          "dated); may be misdated by %d year(s), %s" % (lag, r, rz, abs(lag), cause),
                          series=s, n=int(ok.sum()), value=int(lag)))
    ok = np.isfinite(r0)
    if ok.sum() >= 3:
        med = float(np.median(r0[ok]))
        spread = float(np.median(np.abs(r0[ok] - med)))  # MAD (unscaled, as dplR mad default*? -> use raw)
        if med < ctl["r_cohesion"]:
            out.append(_f("RWL_WEAK_COLLECTION",
                          "the median series correlates with the rest of the collection at r = %.3f, "
                          "below %g; the collection shares little common signal. Normal in some "
                          "ecological collections, not necessarily a fault" % (med, ctl["r_cohesion"]),
                          n=int(ok.sum()), value=med))
        if spread > 0:
            for i in np.where(ok)[0]:
                z = (r0[i] - med) / spread
                if z < -ctl["outlier_mad"] and r0[i] < med - ctl["outlier_gap"]:
                    out.append(_f("RWL_SERIES_OUTLIER",
                                  "correlates with the collection at r = %.3f where the median is %.3f "
                                  "(%.1f MAD below it); it does not fit the collection" % (r0[i], med, abs(z)),
                                  series=cols[i], value=float(r0[i])))
    return out


def _check_file(fname, M, years, cols, ctl):
    out = []
    if not (fname and os.path.isfile(fname)):
        return out
    try:
        raw = open(fname, "rb").read()
        txt = raw.decode("latin1").splitlines()
    except Exception:
        return out
    ncr, nlf = raw.count(b"\r"), raw.count(b"\n")
    if ncr > 0 and nlf > 0 and ncr != nlf:
        out.append(_f("RWL_MIXED_EOL",
                      "line endings are not consistent: %d CR against %d LF; mixed endings can "
                      "defeat a reader" % (ncr, nlf), n=abs(nlf - ncr)))
    ntab = raw.count(b"\t")
    if ntab:
        out.append(_f("RWL_TAB", "%d tab character(s); Tucson columns are fixed width and tabs "
                      "move them" % ntab, n=ntab))
    odd = sum(1 for byte in raw if not (byte in (9, 10, 13) or 32 <= byte <= 126))
    if odd:
        out.append(_f("RWL_NON_ASCII", "%d byte(s) outside printable ASCII" % odd, n=odd))
    if len(raw) and raw[-1:] != b"\n":
        out.append(_f("RWL_NO_FINAL_NEWLINE", "the file does not end with a newline"))
    # ITRDB header span vs data span
    hdr = _itrdb_span(txt)
    if hdr is not None and M.shape[1] > 0:
        d1, d2 = int(years.min()), int(years.max())
        if hdr[0] != d1 or hdr[1] != d2:
            out.append(_f("RWL_HEADER_SPAN",
                          "the header declares %s but the measurements cover %s"
                          % (_span_text(hdr[0], hdr[1]), _span_text(d1, d2)),
                          year_from=d1, year_to=d2,
                          value=float(max(abs(hdr[0] - d1), abs(hdr[1] - d2)))))
    return out


def _itrdb_span(txt):
    """(first, last) year declared on the second ITRDB header line, or None.
    Years may be negative (BC) and short (audit G7: ``-136 2004``)."""
    hdr = [ln for ln in txt[:6] if re.search(r"[A-Za-z]", ln[8:72] if len(ln) > 8 else "")]
    if len(hdr) < 2:
        return None
    m = re.search(r"(-?[0-9]{1,4})\s+(-?[0-9]{1,4})\s*$", hdr[1])
    if not m:
        return None
    y1, y2 = int(m.group(1)), int(m.group(2))
    return (y1, y2) if y1 < y2 else None


def _check_provenance(attrs, cols, ctl):
    out = []
    def count(v):
        try:
            return len(v)
        except TypeError:
            return 1 if v else 0
    if attrs.get("dplpy_encoding_assumed"):
        out.append(_f("RWL_ENCODING_ASSUMED",
                      "the file is not valid UTF-8 and was read as latin1, no encoding declared"))
    if attrs.get("dplpy_mixed_precision"):
        out.append(_f("RWL_MIXED_PRECISION", "the file declares more than one precision"))
    dropped = attrs.get("dplpy_dropped")
    if dropped:
        out.append(_f("RWL_SERIES_DROPPED",
                      "%d series in the file held no readable measurement and is not in the object"
                      % count(dropped), n=count(dropped)))
    for key in ("dplpy_combined", "dplpy_case_merged"):
        val = attrs.get(key)
        if val:
            out.append(_f("RWL_ID_RENAMED",
                          "the reader renamed or merged %d series; the ids on the object are not "
                          "the ids in the file" % count(val), n=count(val)))
    sal = attrs.get("dplpy_salvage")
    if sal:
        out.append(_f("RWL_SALVAGED",
                      "the reader recovered from %d defect(s) on read (salvage mode); see "
                      'df.attrs["dplpy_salvage"]' % count(sal), n=count(sal)))
    beyond = attrs.get("dplpy_beyond_column")
    if beyond:
        # the reader caps the stored list; the exact total is kept alongside
        n_beyond = int(attrs.get("dplpy_beyond_column_n", count(beyond)))
        out.append(_f("RWL_SALVAGED",
                      "%d line(s) carried characters beyond the last data column" % n_beyond,
                      n=n_beyond))
    return out


_GROUPS = {"structure": _check_structure, "series": _check_series,
           "values": _check_values, "zeros": _check_zeros,
           "crossdating": _check_crossdating}


def _check_one(x, file, checks, ctl):
    attrs = {}
    read_error = None
    if isinstance(x, str):
        if file is None:
            file = x
        from .readers import readers
        try:
            with warnings.catch_warnings():
                # silence only the reader diagnostics check() re-reports as findings
                # (a warnings filter regex must match from the START of the message)
                warnings.filterwarnings("ignore", message=_READER_NOISE.pattern,
                                        category=UserWarning)
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rwl = readers(x, strict=False)
            if rwl is None:
                raise ValueError("no usable data could be read")
            attrs = dict(getattr(rwl, "attrs", {}))
        except Exception as e:
            read_error = str(e)
            rwl = None
    else:
        rwl = _require_dataframe(x)
        attrs = dict(getattr(rwl, "attrs", {}))
        if file is None:
            file = "<dataframe>"
    if rwl is not None:
        rwl = _require_year_index(rwl, "x")

    rows = []
    if read_error is not None:
        rows.append(_f("RWL_CHECK_ERROR", "the file could not be read: " + read_error))
        M = np.empty((0, 0))
        years = np.array([])
        cols = []
    else:
        cols = list(rwl.columns)
        M = rwl.to_numpy(dtype=float)
        years = np.asarray(rwl.index, dtype=float)
        for nm in checks:
            if nm in _GROUPS:
                try:
                    rows += _GROUPS[nm](M, years, cols, ctl)
                except Exception as e:
                    rows.append(_f("RWL_CHECK_ERROR", "the %s checks failed to run: %s" % (nm, e)))
        if "provenance" in checks:
            try:
                rows += _check_provenance(attrs, cols, ctl)
            except Exception as e:
                rows.append(_f("RWL_CHECK_ERROR", "the provenance checks failed to run: %s" % e))

    if "file" in checks and isinstance(x, str):
        try:
            rows += _check_file(x, M, years, cols, ctl)
        except Exception as e:
            rows.append(_f("RWL_CHECK_ERROR", "the file checks failed to run: %s" % e))

    for r in rows:
        r["severity"] = _SEV.get(r["check"], "note")
        r["group"] = _GRP.get(r["check"], "engine")
        r["file"] = file
    df = pd.DataFrame(rows, columns=_FINDING_COLS) if rows else pd.DataFrame(columns=_FINDING_COLS)
    if len(df):
        df["_s"] = df["severity"].map(_SEV_ORDER).fillna(3)
        df = df.sort_values(["_s", "check", "series", "year_from"]).drop(columns="_s").reset_index(drop=True)
    return df


def check(x, file=None, checks=None, *, min_depth: int = 2, depth_run: int = 10,
          min_length: int = 30, run_min: int = 6, lag: int = 10,
          min_overlap: int = 50, spline_period: int = 32, r_dating: float = 0.35,
          r_margin: float = 0.05, outlier_mad: float = 4.0, outlier_gap: float = 0.2,
          r_cohesion: float = 0.35, plausible_mean: tuple = (0.05, 10.0),
          small_thresh: Optional[float] = None, big_thresh: Optional[float] = None,
          max_year: Optional[int] = None, summary: bool = False, **kwargs):
    """Check a ring-width collection for defects, reported as a table of findings.

    The data counterpart to `report()`: ``report`` prints what a file looks
    like, ``check`` returns what is wrong with it, one row per finding. Every
    finding carries a stable ``check`` id and a ``severity`` ("error", "warning",
    "note"). Analog of dplR 1.8.0's ``rwl.check``.

    The crossdating group slides the leave-one-out master past each series over
    ``-lag..+lag`` (`xdate._lag_table()`), so a ``RWL_DATING_LAG`` with a
    NEGATIVE lag means a MISSING ring in the series, as in COFECHA, ``xdate`` and
    ``series_corr``.

    Parameters
    ----------
    x : pandas.DataFrame, str, or list of str
        a ring-width collection (``readers`` output), a path to a Tucson file, or a
        list of paths / a directory for an archive sweep.
    file : str or None
        label for the ``file`` column when ``x`` is a DataFrame.
    checks : list of str or None
        which groups to run. Default all: structure, series, values, zeros,
        crossdating, file, provenance.
    summary : bool, default False
        if True, return one row per file (counts per check and per severity) instead
        of the findings. ``pd.concat`` these over an archive to rank it.
    min_depth, depth_run, min_length, run_min, lag, min_overlap, spline_period,
    r_dating, r_margin, outlier_mad, outlier_gap, r_cohesion, plausible_mean,
    small_thresh, big_thresh, max_year :
        tunables; defaults follow dplR's ``rwl.check.control`` except ``lag`` (10,
        dplPy's package-wide lag search) and ``spline_period`` (32, the high-pass
        spline). ``lag_max=`` and ``spline_nyrs=`` are accepted as deprecated
        aliases.

    Returns
    -------
    pandas.DataFrame
        the findings (columns ``file, check, severity, group, series, year_from,
        year_to, n, value, message``), or, with ``summary=True``, one row per file.

    Notes
    -----
    Every check group runs inside a guard: an exception inside a group (or in
    reading the file) is converted to a ``RWL_CHECK_ERROR`` finding carrying the
    error text, so one failing check never hides the others' findings and a sweep
    over an archive never stops on one bad file (audit G4). Check the ``engine``
    group of the result to see whether anything failed to run. A misdated series
    may draw both ``RWL_DATING_LAG`` and ``RWL_SERIES_OUTLIER`` (its lag-0
    correlation is low because of the lag); both are kept.

    References
    ----------
    .. [1] Bunn, A. G. (2026) dplR 1.8.0, ``rwl.check``.
    """
    lag = _alias_kwarg(kwargs, "lag_max", "lag", lag, "check")
    spline_period = _alias_kwarg(kwargs, "spline_nyrs", "spline_period", spline_period,
                                 "check")
    _reject_kwargs(kwargs)
    if max_year is None:
        max_year = date.today().year
    all_groups = ["structure", "series", "values", "zeros", "crossdating",
                  "file", "provenance"]
    if checks is None:
        checks = all_groups
    else:
        bad = [c for c in checks if c not in all_groups]
        if bad:
            raise ValueError("unknown check group(s): " + ", ".join(bad))
    ctl = dict(min_depth=min_depth, depth_run=depth_run, min_length=min_length,
               run_min=run_min, lag=int(lag), min_overlap=min_overlap,
               spline_period=int(spline_period), r_dating=r_dating, r_margin=r_margin,
               outlier_mad=outlier_mad, outlier_gap=outlier_gap, r_cohesion=r_cohesion,
               plausible_mean=plausible_mean, small_thresh=small_thresh,
               big_thresh=big_thresh, max_year=int(max_year))

    # archive sweep: a directory or a list of paths
    if isinstance(x, str) and os.path.isdir(x):
        files = sorted(sum([glob.glob(os.path.join(x, "*" + e))
                            for e in (".rwl", ".txt", ".crn", ".raw")], []))
        x = files
    if isinstance(x, (list, tuple)):
        parts = [check(xi, checks=checks, summary=summary, min_depth=min_depth,
                       depth_run=depth_run, min_length=min_length, run_min=run_min,
                       lag=lag, min_overlap=min_overlap, spline_period=spline_period,
                       r_dating=r_dating, r_margin=r_margin, outlier_mad=outlier_mad,
                       outlier_gap=outlier_gap, r_cohesion=r_cohesion,
                       plausible_mean=plausible_mean, small_thresh=small_thresh,
                       big_thresh=big_thresh, max_year=max_year) for xi in x]
        if not parts:
            return pd.DataFrame(columns=_FINDING_COLS)
        return pd.concat(parts, ignore_index=True)

    findings = _check_one(x, file, checks, ctl)
    if not summary:
        return findings
    return _summarize(findings, x, file)


def _summarize(findings, x, file):
    """One row per file: span/size meta plus counts per severity and per check."""
    fname = file if file is not None else (x if isinstance(x, str) else "<dataframe>")
    row = {"file": fname}
    for sev in ("error", "warning", "note"):
        row["n_" + sev] = int((findings["severity"] == sev).sum())
    for cid, _, _, _ in _CATALOGUE:
        row[cid] = int((findings["check"] == cid).sum())
    return pd.DataFrame([row])
