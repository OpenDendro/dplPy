"""Fire-history series/site statistics and compositing.

A port of the series-statistics and compositing layer of the R package burnr
(Malevich, Guiterman & Margolis; Malevich et al. 2018, Dendrochronologia,
doi:10.1016/j.dendro.2018.02.005), operating on the ``fhx`` event frames built by
:mod:`dplpy.fhx`.

Public functions mirror burnr's verbs: :func:`series_stats` and its per-series
pieces (:func:`first_year`, :func:`last_year`, :func:`count_year_span`,
:func:`inner_type`, :func:`outer_type`, :func:`count_scar`, :func:`count_injury`,
:func:`count_recording`, :func:`series_mean_interval`), :func:`sample_depth`,
:func:`percent_scarred`, :func:`yearly_recording`, :func:`summary_fhx`, and
:func:`composite`.

The "recording" rule (:func:`_find_recording`) is ported verbatim from burnr's
internal ``find_recording`` so that recording counts, composites, and
percent-scarred series match burnr exactly.
"""
import re

import numpy as np
import pandas as pd

from ._fhx_vocab import RECORDER, INJURY, SCAR, ENDS
from .fhx import (_require_fhx, series_names, get_event_years, fhx,
                  make_rec_type)


# --- per-series scalar pieces ---------------------------------------------

def first_year(x):
    """Earliest year in an ``fhx`` object (or one of its series)."""
    _require_fhx(x)
    return int(x["year"].min())


def last_year(x):
    """Latest year in an ``fhx`` object (or one of its series)."""
    _require_fhx(x)
    return int(x["year"].max())


def count_year_span(x):
    """Number of years spanned, ``last_year - first_year + 1``."""
    _require_fhx(x)
    return int(x["year"].max() - x["year"].min() + 1)


def inner_type(x):
    """``rec_type`` (a string) of the earliest observation in ``x``."""
    _require_fhx(x)
    return str(x["rec_type"].iloc[int(np.asarray(x["year"]).argmin())])


def outer_type(x):
    """``rec_type`` (a string) of the latest observation in ``x``."""
    _require_fhx(x)
    return str(x["rec_type"].iloc[int(np.asarray(x["year"]).argmax())])


def count_scar(x):
    """Number of fire-scar (``_fs``) events in ``x``."""
    _require_fhx(x)
    return int(x["rec_type"].astype(str).str.contains("_fs").sum())


def count_injury(x):
    """Number of injury (``_fi``) events in ``x``."""
    _require_fhx(x)
    return int(x["rec_type"].astype(str).str.contains("_fi").sum())


def count_recording(x, injury_event=False):
    """Number of recording years in ``x`` (see :func:`_find_recording`)."""
    _require_fhx(x)
    return int(len(_find_recording(x, injury_event=injury_event)))


def series_mean_interval(x, injury_event=False):
    """Quick mean fire interval of a single-series ``fhx`` object.

    The mean of the gaps between successive scar years (plus injury years if
    ``injury_event``), rounded to one decimal. Returns ``nan`` with fewer than
    two events. Prefer :func:`dplpy.intervals` for real interval analysis.
    """
    _require_fhx(x)
    if len(series_names(x)) > 1:
        from ._validate import _warn
        _warn("series_mean_interval run on an object with multiple series; "
              "results may not be meaningful.")
    pat = ("_fi|_fs" if injury_event else "_fs")
    rt = x["rec_type"].astype(str)
    event_years = np.sort(x.loc[rt.str.contains(pat, regex=True), "year"]
                          .to_numpy())
    if len(event_years) > 1:
        return _r_round(float(np.diff(event_years).mean()), 1)
    return float("nan")


# --- the recording rule (verbatim port of burnr find_recording) -----------

def _find_recording(x, injury_event=False):
    """Years considered "recording" in ``x`` (generally one series).

    Verbatim port of burnr's internal ``find_recording``. A recording year is a
    recorder year or scar (and injuries too when ``injury_event``); a
    pith/bark/inner/outer "end" or an injury counts as recording when it sits
    immediately beside a recording year. Returns a sorted unique ``int`` array.
    """
    yrs = x["year"].to_numpy()
    rt = x["rec_type"].astype(str).to_numpy()

    recorder = list(RECORDER)
    if injury_event:
        recorder = recorder + list(INJURY)

    rec = np.unique(yrs[np.isin(rt, recorder)])
    inj = np.unique(yrs[np.isin(rt, list(INJURY))])
    end = np.unique(yrs[np.isin(rt, list(ENDS))])

    rec_set = set(int(v) for v in rec)
    end_set = set(int(v) for v in end)
    inj_set = set(int(v) for v in inj)

    # ends adjacent to a recording year are "active" (recording)
    active = set(rec_set)
    active |= {v for v in end_set if (v + 1) in rec_set}   # intersect(rec-1, end)
    active |= {v for v in end_set if (v - 1) in rec_set}   # intersect(rec+1, end)
    # injuries adjacent to an active year are recording too
    adj = {v for v in inj_set if (v + 1) in active} | \
          {v for v in inj_set if (v - 1) in active}
    active |= adj

    # burnr also carries recording-ness across runs of consecutive injuries.
    # This guard tests the value 1 (a diff) against the year set, so for any real
    # calendar data it is inert; kept here only for faithfulness to burnr.
    inj_sorted = np.sort(np.array(sorted(inj_set), dtype="int64")) \
        if inj_set else np.array([], dtype="int64")
    inj_dif = np.diff(inj_sorted)
    if (not injury_event) and np.any(inj_dif == 1):
        for i in np.where(inj_dif == 1)[0]:
            if int(inj_dif[i]) in active and i + 1 < len(inj_dif):
                active.add(int(inj_dif[i + 1]))

    return np.array(sorted(rec_set | active), dtype="int64")


# --- series-level table ---------------------------------------------------

def series_stats(x):
    """Per-series descriptive statistics for an ``fhx`` object.

    Returns a DataFrame with one row per series (in first-appearance order) and
    columns ``series``, ``first``, ``last``, ``years``, ``inner_type``,
    ``outer_type``, ``number_scars``, ``number_injuries``, ``recording_years``,
    ``mean_interval`` -- mirroring burnr's :func:`series_stats`.
    """
    _require_fhx(x)
    rows = []
    for s in series_names(x):
        df = x[x["series"].astype(str) == s]
        rows.append({
            "series": s,
            "first": first_year(df),
            "last": last_year(df),
            "years": count_year_span(df),
            "inner_type": inner_type(df),
            "outer_type": outer_type(df),
            "number_scars": count_scar(df),
            "number_injuries": count_injury(df),
            "recording_years": count_recording(df),
            "mean_interval": series_mean_interval(df),
        })
    return pd.DataFrame(rows, columns=[
        "series", "first", "last", "years", "inner_type", "outer_type",
        "number_scars", "number_injuries", "recording_years", "mean_interval"])


def summary_fhx(x):
    """Brief summary of an ``fhx`` object as a dict: number of series, first and
    last year, and total scar and injury counts (burnr's ``summary.fhx``)."""
    _require_fhx(x)
    return {
        "number_series": len(series_names(x)),
        "first_year": first_year(x),
        "last_year": last_year(x),
        "number_scars": count_scar(x),
        "number_injuries": count_injury(x),
    }


# --- sample depth ----------------------------------------------------------

def sample_depth(x):
    """Sample depth of an ``fhx`` object as a DataFrame of ``year`` and
    ``samp_depth`` -- the number of series whose first-to-last span covers each
    year, over the full year range."""
    _require_fhx(x)
    ss = series_stats(x)
    years = np.arange(int(ss["first"].min()), int(ss["last"].max()) + 1)
    depth = np.zeros(len(years), dtype="int64")
    base = years[0]
    for first, last in zip(ss["first"].to_numpy(), ss["last"].to_numpy()):
        depth[int(first) - base: int(last) - base + 1] += 1
    return pd.DataFrame({"year": years, "samp_depth": depth})


# --- per-year recording and percent-scarred -------------------------------

def yearly_recording(x, injury_event=False):
    """Count of recording series in each year, as a DataFrame of ``year`` and
    ``Freq`` (burnr's :func:`yearly_recording`)."""
    _require_fhx(x)
    all_years = []
    for s in series_names(x):
        df = x[x["series"].astype(str) == s]
        all_years.append(_find_recording(df, injury_event=injury_event))
    if all_years:
        all_years = np.concatenate(all_years)
    else:
        all_years = np.array([], dtype="int64")
    vals, counts = np.unique(all_years, return_counts=True)
    return pd.DataFrame({"year": vals.astype("int64"),
                         "Freq": counts.astype("int64")})


def percent_scarred(x, injury_event=False):
    """Percent-scarred time series for an ``fhx`` object.

    Returns a DataFrame with columns ``year``, ``num_rec`` (recording trees),
    ``num_scars`` (scars, plus injuries if ``injury_event``), and
    ``percent_scarred`` (``round(num_scars / num_rec * 100)``). As in burnr, the
    recording denominator always counts injuries as recording.
    """
    _require_fhx(x)
    # recording count per year -- burnr always passes injury_event=TRUE here
    rec_years = []
    for s in series_names(x):
        df = x[x["series"].astype(str) == s]
        rec_years.append(_find_recording(df, injury_event=True))
    rec_years = (np.concatenate(rec_years) if rec_years
                 else np.array([], dtype="int64"))
    rvals, rcounts = np.unique(rec_years, return_counts=True)
    rec_count = pd.DataFrame({"year": rvals.astype("int64"),
                              "num_rec": rcounts.astype("int64")})

    pat = ("_fs|_fi" if injury_event else "_fs")
    rt = x["rec_type"].astype(str)
    fs_years = x.loc[rt.str.contains(pat, regex=True), "year"].to_numpy()
    fvals, fcounts = np.unique(fs_years, return_counts=True)
    fs_count = pd.DataFrame({"year": fvals.astype("int64"),
                             "num_scars": fcounts.astype("int64")})

    out = rec_count.merge(fs_count, on="year", how="outer").sort_values("year")
    out["num_scars"] = out["num_scars"].fillna(0).astype("int64")
    out["percent_scarred"] = [
        _r_round(ns / nr * 100.0, 0) if pd.notna(nr) else float("nan")
        for ns, nr in zip(out["num_scars"], out["num_rec"])
    ]
    return out.reset_index(drop=True)


# --- compositing -----------------------------------------------------------

def composite(x, filter_prop=0.25, filter_min_rec=2, filter_min_events=1,
              injury_event=False, comp_name="COMP"):
    """Composite fire record of an ``fhx`` object.

    A year is a composite fire year when the fraction of recording trees that
    scarred is at least ``filter_prop``, at least ``filter_min_rec`` trees were
    recording, and at least ``filter_min_events`` events occurred. Returns a
    one-series ``fhx`` (scar years as ``unknown_fs``, the span's ends as
    inner/outer years, and the years after the first event as recorder years),
    or an empty ``fhx`` if no year qualifies. Mirrors burnr's :func:`composite`.
    """
    _require_fhx(x)
    event = list(SCAR) + (list(INJURY) if injury_event else [])
    rt = x["rec_type"].astype(str)
    event_year = x.loc[rt.isin(event), "year"].to_numpy()
    if len(event_year) < 1:
        return _empty_fhx()

    evals, ecounts = np.unique(event_year, return_counts=True)
    event_count = pd.DataFrame({"year": evals.astype("int64"),
                                "Freq_event": ecounts.astype("int64")})
    recording_count = yearly_recording(x, injury_event=injury_event).rename(
        columns={"Freq": "Freq_recording"})

    counts = event_count.merge(recording_count, on="year", how="inner")
    prop = counts["Freq_event"] / counts["Freq_recording"]
    mask = ((prop >= filter_prop)
            & (counts["Freq_recording"] >= filter_min_rec)
            & (counts["Freq_event"] >= filter_min_events))
    comp_years = np.sort(counts.loc[mask, "year"].to_numpy().astype("int64"))

    if len(comp_years) == 0:
        return _empty_fhx()

    xmin, xmax = int(x["year"].min()), int(x["year"].max())
    out_year = list(comp_years) + [xmin, xmax]
    out_rec = (["unknown_fs"] * len(comp_years) + ["inner_year", "outer_year"])
    used = set(out_year)
    recorder = [y for y in range(int(comp_years.min()), xmax + 1)
                if y not in used]
    out_year += recorder
    out_rec += ["recorder_year"] * len(recorder)
    out_series = [str(comp_name)] * len(out_year)
    return fhx(out_year, out_series, make_rec_type(out_rec))


def _empty_fhx():
    out = fhx([], [], make_rec_type([]))
    return out


# --- rounding --------------------------------------------------------------

def _r_round(value, digits=0):
    """Round half to even, matching R's ``round`` (and NumPy), returning a float
    (``digits>0``) or an int-valued float (``digits==0``)."""
    if not np.isfinite(value):
        return float("nan")
    return float(np.round(value, digits))
