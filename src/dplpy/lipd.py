__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

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

# Title: lipd.py
# Description: LiPD (Linked Paleo Data) export for dplPy, delegating all of the
#   container/JSON-LD/BagIt machinery to `pylipd` (an optional dependency:
#   `pip install "dplpy[lipd]"`). dplPy maps its outputs onto pylipd's object model.
#
#   Design (see dev/lipd_integration_proposal.md, decision 10.4): the CHRONOLOGY
#   is written as the *primary* paleoData measurement table, so the standard
#   pylipd/Pyleoclim extraction (`get_timeseries`) returns it directly. Raw ring
#   widths, when supplied, go in a further measurement table. Additional
#   chronologies (residual, ARSTAN, ...) are written as their own measurement
#   tables. pylipd's `get_timeseries` normalises the chronology's `trsgi` to the
#   LiPD ontology umbrella name `ringWidth`; the stored variableName is `trsgi` and
#   `get_all_variable_names` reports `trsgi`, so the chronology stays identifiable.
#
# example usage:
#   >>> import dplpy as dpl
#   >>> rwl = dpl.readers("ca533.rwl")
#   >>> rwi = dpl.detrend(rwl, fit="Spline", method="ratio")
#   >>> # a single standard chronology:
#   >>> dpl.to_lipd(dpl.chron(rwi), "ca533.lpd", rwl=rwl)
#   >>> # standard + residual + ARSTAN, with running signal statistics + a citation:
#   >>> ars  = dpl.chron_ars(rwi)
#   >>> stab = dpl.chron_stabilized(rwi, running_rbar=True)
#   >>> dpl.to_lipd(ars, "ca533.lpd", rwl=rwl, chronologies="all", stats=stab,
#   ...             provenance="cubic smoothing spline detrend; biweight robust mean",
#   ...             publication={"authors": "Lamarche, V.C.", "year": 1991,
#   ...                          "doi": "10.xxxx/xxxxx"})

import os
import re
import json

import pandas as pd

from ._lipd_support import require_pylipd
from .site_metadata import SiteMetadata
from .readers import _METADATA_FIELDS
from . import lipd_vocab

# Human-readable labels for dplPy chronology columns (used in descriptions and
# in the deterministic TSids so several chronologies in one file stay distinct).
_CHRON_LABELS = {"std": "standard", "res": "residual", "ars": "ARSTAN",
                 "vsc": "variance-stabilized", "sfc": "signal-free"}


def _sanitize(text) -> str:
    """A filesystem/id-safe token: keep alphanumerics, collapse the rest to '-'."""
    return re.sub(r"[^0-9A-Za-z]+", "-", str(text)).strip("-") or "series"


def _tsid(stem: str, *parts) -> str:
    """A deterministic, stable TSid (decision 10.3): a re-export of the same
    chronology reuses the same ids, so a record updates rather than forks."""
    return "-".join([stem, *[_sanitize(p) for p in parts]])


def _attrs(varkey: str, number: int, tsid: str, extra: dict = None) -> dict:
    """Per-column LiPD variable metadata (rides in df.attrs; consumed by pylipd's
    DataTable.setDataFrame). Pulls the controlled variableName/units from
    lipd_vocab.

    We intentionally do NOT emit ``hasStandardVariable``: pylipd resolves it
    against the LiPD ontology and *overwrites* the stored ``variableName`` with the
    ontology's umbrella term -- which for ``trsgi`` is ``ringWidth``, making a
    chronology indistinguishable from raw ring widths. Omitting it keeps our
    specific term (``trsgi``) as the stored ``variableName`` (and what
    ``get_all_variable_names`` reports); ``get_timeseries`` still normalises to the
    ontology name at read time."""
    spec = lipd_vocab.lipd_variable(varkey)
    d = {"variableName": spec["variableName"], "units": spec["units"],
         "number": number, "TSid": tsid}
    if extra:
        d.update(extra)
    return d


def _resolve_chronologies(chron, column, chronologies):
    """Return an ordered list of (col, label) chronology columns to write, primary
    first. ``chronologies`` may be None (just ``column``), 'all' (every non
    sample-depth column), or an explicit list of column names."""
    if chronologies is None:
        cols = [column]
    elif chronologies == "all":
        cols = [c for c in chron.columns if c != "samp_depth"]
        # make sure the requested primary column leads
        if column in cols:
            cols = [column] + [c for c in cols if c != column]
    else:
        cols = list(chronologies)
    missing = [c for c in cols if c not in chron.columns]
    if missing:
        raise ValueError("chronology column(s) not found: " + ", ".join(missing)
                         + "; available: " + str(list(chron.columns)))
    return [(c, _CHRON_LABELS.get(c, c)) for c in cols]


def _running_stats(stats, years, count):
    """From an optional per-year stats frame (e.g. chron_stabilized output),
    return (rbar_series, eps_series) aligned to ``years``, or (None, None).
    Recognises a running-rbar column by name; derives EPS from rbar and sample
    depth if an EPS column is not present."""
    if stats is None:
        return None, None
    import numpy as np
    idx = pd.Index([int(y) for y in stats.index])
    s = stats.copy()
    s.index = idx
    def _find(sub):
        for c in s.columns:
            if sub in str(c).lower().replace(" ", "").replace("_", ""):
                return s[c]
        return None
    rbar = _find("rbar")
    eps = _find("eps")
    yidx = pd.Index([int(y) for y in years])
    rbar_v = rbar.reindex(yidx).to_numpy(dtype=float) if rbar is not None else None
    if eps is not None:
        eps_v = eps.reindex(yidx).to_numpy(dtype=float)
    elif rbar_v is not None and count is not None:
        n = pd.Series(count).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            eps_v = (n * rbar_v) / (n * rbar_v + (1.0 - rbar_v))
    else:
        eps_v = None
    return rbar_v, eps_v


def _chron_table(DataTable, chron, col, label, stem, primary, provenance,
                 stats=None):
    """Build one chronology measurement DataTable: year, trsgi, [count],
    [RBAR, EPS on the primary table]."""
    years = [int(y) for y in chron.index]
    tag = label if label else col
    cdf = pd.DataFrame({"year": years, "trsgi": chron[col].to_numpy(dtype=float)})
    desc = tag.capitalize() + " chronology built with dplPy"
    if provenance:
        desc = desc + " (" + str(provenance) + ")"
    trsgi_extra = {"isPrimary": bool(primary), "description": desc,
                   "longName": lipd_vocab.past_long_name(
                       "standardized growth index", "dimensionless",
                       detail=(str(provenance) if provenance else ""),
                       method=tag + " chronology")}
    if provenance:
        trsgi_extra["notes"] = str(provenance)
    cattrs = {
        "year": _attrs("year", 1, _tsid(stem, tag, "year")),
        "trsgi": _attrs("trsgi", 2, _tsid(stem, tag, "trsgi"), trsgi_extra),
    }
    number = 3
    count = None
    if "samp_depth" in chron.columns:
        count = chron["samp_depth"].to_numpy(dtype=float)
        cdf["count"] = count
        cattrs["count"] = _attrs("sample_count", number, _tsid(stem, tag, "sampleCount"))
        number += 1
    if primary and stats is not None:
        rbar_v, eps_v = _running_stats(stats, years, count)
        if rbar_v is not None:
            cdf["RBAR"] = rbar_v
            cattrs["RBAR"] = _attrs("rbar", number, _tsid(stem, tag, "RBAR")); number += 1
        if eps_v is not None:
            cdf["EPS"] = eps_v
            cattrs["EPS"] = _attrs("eps", number, _tsid(stem, tag, "EPS")); number += 1
    cdf.attrs = cattrs
    t = DataTable()
    t.setMissingValue("NaN")
    t.setDataFrame(cdf)
    return t


def _widths_table(DataTable, rwl, stem):
    """Build the raw-ring-width measurement DataTable (year + one ringWidth per
    series)."""
    series = [str(c) for c in rwl.columns]
    mdf = rwl.reset_index()
    mdf.columns = ["year", *series]
    mattrs = {"year": _attrs("year", 1, _tsid(stem, "rw", "year"))}
    for i, s in enumerate(series, start=2):
        mattrs[s] = _attrs("ring_width", i, _tsid(stem, s, "ringWidth"),
                           {"description": "Raw ring width measurements from series "
                                           + s, "longName": s + " ring width"})
    mdf.attrs = mattrs
    t = DataTable()
    t.setMissingValue("NaN")
    t.setDataFrame(mdf)
    return t


def _persons(Person, authors):
    """Coerce an author string ('A; B') or list into a list of pylipd Person."""
    if isinstance(authors, str):
        names = [a.strip() for a in re.split(r";", authors) if a.strip()] or [authors]
    else:
        names = [str(a) for a in authors]
    out = []
    for n in names:
        p = Person(); p.setName(n); out.append(p)
    return out


def _build_publication(Publication, Person, publication):
    """Build a pylipd Publication from a dict (authors/author, doi, title, year,
    journal, url, citation), or None."""
    if not publication:
        return None
    pub = Publication()
    authors = publication.get("authors") or publication.get("author")
    if authors:
        try:
            pub.setAuthors(_persons(Person, authors))
        except Exception:                                    # pragma: no cover
            pub.set_non_standard_property("author", str(authors))
    for key, setter in (("doi", "setDOI"), ("title", "setTitle"),
                        ("journal", "setJournal"), ("citation", "setCitation")):
        if publication.get(key) is not None:
            getattr(pub, setter)(str(publication[key]))
    if publication.get("year") is not None:
        try:
            pub.setYear(int(publication["year"]))
        except Exception:                                    # pragma: no cover
            pub.set_non_standard_property("year", str(publication["year"]))
    url = publication.get("url") or publication.get("dataUrl")
    if url:
        try:
            pub.setUrls([str(url)])
        except Exception:                                    # pragma: no cover
            pub.set_non_standard_property("url", str(url))
    return pub


def to_lipd(chron, filename, rwl=None, metadata=None, dsname=None,
            column="std", chronologies=None, stats=None,
            publication=None, provenance=None, archive_type=None):
    """Write dplPy chronologies to a LiPD ``.lpd`` file.

    Parameters
    ----------
    chron : pandas.DataFrame
        Chronology output from ``dpl.chron()`` or ``dpl.chron_ars()`` -- year
        indexed, with one or more chronology columns and, if present,
        ``'samp_depth'`` (written as the ``count`` / ``sampleCount`` companion).
    filename : str
        Output path. ``.lpd`` is appended if missing.
    rwl : pandas.DataFrame, optional
        Raw ring widths the chronology was built from. Written as a further
        measurement table, and -- if ``metadata`` is None -- used to auto-fill site
        metadata from its ``dplpy_metadata`` attrs.
    metadata : SiteMetadata or dict, optional
        Site metadata for the ``geo`` block; defaults to what ``rwl`` carries.
    dsname : str, optional
        LiPD dataset name; defaults to the filename stem (a purely-numeric name is
        avoided -- pylipd cannot resolve one).
    column : str, default 'std'
        The primary chronology column (written with ``isPrimary=True``).
    chronologies : None | 'all' | list of str, default None
        Which chronology columns to write. None writes just ``column``; ``'all'``
        writes every non ``samp_depth`` column (e.g. std/res/ars from
        ``chron_ars``); a list writes those columns. The primary is written first.
    stats : pandas.DataFrame, optional
        Per-year running statistics (e.g. ``dpl.chron_stabilized(..., running_rbar
        =True)``); a running-rbar column is written as ``RBAR`` on the primary
        chronology table and EPS is derived from it and sample depth (or taken from
        an EPS column if present).
    publication : dict, optional
        Citation for the record: any of ``authors``/``author`` (str or list),
        ``doi``, ``title``, ``journal``, ``year``, ``url``, ``citation``.
    provenance : str, optional
        Free-text description of how the chronology was built (detrending, AR,
        robust mean, ...). Stored on each chronology variable's description/notes
        and PaST long name.
    archive_type : str, optional
        LiPD archiveType. Defaults to ``'Wood'``.

    Returns
    -------
    str
        The path written.

    Notes
    -----
    Requires the optional ``pylipd`` dependency: ``pip install "dplpy[lipd]"``.
    """
    require_pylipd()
    from pylipd import LiPD
    from pylipd.classes.dataset import Dataset
    from pylipd.classes.paleodata import PaleoData
    from pylipd.classes.datatable import DataTable
    from pylipd.classes.location import Location
    from pylipd.classes.publication import Publication
    from pylipd.classes.person import Person

    if not isinstance(chron, pd.DataFrame):
        raise TypeError("chron must be a pandas DataFrame from dpl.chron() / "
                        "dpl.chron_ars(), not " + str(type(chron)))
    if rwl is not None and not isinstance(rwl, pd.DataFrame):
        raise TypeError("rwl must be a pandas DataFrame of ring widths or None.")
    chron_cols = _resolve_chronologies(chron, column, chronologies)
    primary_col = chron_cols[0][0]

    # -- site metadata (auto from rwl if not given) ------------------------- #
    if metadata is not None:
        meta = SiteMetadata.coerce(metadata)
    elif isinstance(rwl, pd.DataFrame) and rwl.attrs.get("dplpy_metadata"):
        meta = SiteMetadata.from_rwl(rwl)
    else:
        meta = SiteMetadata()

    if not str(filename).lower().endswith(".lpd"):
        filename = str(filename) + ".lpd"
    file_stem = os.path.splitext(os.path.basename(filename))[0]
    dsname = str(dsname or file_stem or "dataset").strip()
    if re.fullmatch(r"[0-9.]+", dsname) or not dsname:   # numeric names break pylipd
        dsname = "dplpy-" + (dsname.replace(".", "-") or "dataset")
    stem = _sanitize(dsname)

    # -- measurement tables: chronologies (primary first) then raw widths --- #
    tables = []
    for i, (col, label) in enumerate(chron_cols):
        tables.append(_chron_table(DataTable, chron, col, label, stem,
                                   primary=(i == 0), provenance=provenance,
                                   stats=stats if i == 0 else None))
    if isinstance(rwl, pd.DataFrame):
        tables.append(_widths_table(DataTable, rwl, stem))

    # sequential LiPD filenames; the primary chronology is measurement1
    for n, t in enumerate(tables, start=1):
        t.setFileName(stem + ".paleo1measurement" + str(n) + ".csv")

    pdo = PaleoData()
    pdo.setName("paleo1")
    pdo.setMeasurementTables(tables)

    ds = Dataset()
    ds.setName(dsname)
    ds.setPaleoData([pdo])
    ds.set_non_standard_property("archiveType", archive_type or lipd_vocab.ARCHIVE_TYPE)

    # -- geo -------------------------------------------------------------- #
    loc = Location()
    _set = lambda setter, value: setter(str(value)) if value is not None else None
    _set(loc.setLatitude, meta.latitude)
    _set(loc.setLongitude, meta.longitude)
    _set(loc.setElevation, meta.elevation_m)
    _set(loc.setSiteName, meta.site_name)
    _set(loc.setLocationName, meta.country_region)
    if getattr(meta, "site_id", None):
        loc.set_non_standard_property("itrdbName", str(meta.site_id))
    ds.setLocation(loc)

    # -- investigators + publication -------------------------------------- #
    if getattr(meta, "investigators", None):
        try:
            ds.setInvestigators(_persons(Person, meta.investigators))
        except Exception:                                    # pragma: no cover
            pass
    pub = _build_publication(Publication, Person, publication)
    if pub is not None:
        ds.setPublications([pub])

    lpd = LiPD()
    lpd.load_datasets([ds])
    lpd.create_lipd(dsname, filename)
    return filename


# --------------------------------------------------------------------------- #
# read side (Phase 3): LiPD .lpd -> dplPy-native frames
# --------------------------------------------------------------------------- #
# The chronology may live in a primary measurement table (dplPy's own layout) or
# in a model summary table (the ITRDB/Nam2k layout); from_lipd finds it in either
# place. All parsing is delegated to pylipd; we map its object model to dplPy
# frames. pylipd returns a variable's values as a JSON string, and does not
# preserve table/column order, so columns are identified by name, not position.
def _values(var):
    """A pylipd Variable's values as a Python list (getValues() is a JSON string)."""
    raw = var.getValues()
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:                                    # pragma: no cover
            return list(raw)
    return list(raw) if raw is not None else []


def _year_var(variables):
    for v in variables:
        if str(v.getName()).lower() in ("year", "age"):
            return v
    return None


def _chron_label(desc, fallback):
    """First word of a chronology's description ('Standard'/'Residual'/'ARSTAN')
    lower-cased, else a fallback label."""
    if desc:
        words = str(desc).strip().split()
        if words:
            return words[0].lower()
    return fallback


def _all_tables(ds):
    """Every table in a dataset's paleoData: measurement tables first, then model
    summary tables (so an ITRDB-layout chronology in a model summary is found)."""
    tables = []
    for pdo in (ds.getPaleoData() or []):
        for mt in (pdo.getMeasurementTables() or []):
            tables.append(mt)
        for mdl in (pdo.getModeledBy() or []):
            for st in (mdl.getSummaryTables() or []):
                tables.append(st)
    return tables


def _read_metadata(ds):
    """Build a dpl.metadata()-shape dict from a LiPD dataset's geo + people."""
    md = {k: None for k in _METADATA_FIELDS}

    def _get(obj, name):
        try:
            return getattr(obj, name)()
        except Exception:                                    # pragma: no cover
            return None

    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return x

    loc = _get(ds, "getLocation")
    if loc is not None:
        md["latitude"] = _num(_get(loc, "getLatitude"))
        md["longitude"] = _num(_get(loc, "getLongitude"))
        md["elevation_m"] = _num(_get(loc, "getElevation"))
        md["site_name"] = _get(loc, "getSiteName")
        md["country_region"] = _get(loc, "getLocationName")
        try:
            md["site_id"] = loc.get_non_standard_property("itrdbName")
        except Exception:
            pass
    invs = _get(ds, "getInvestigators")
    if invs:
        names = [p.getName() for p in invs if getattr(p, "getName", None)]
        md["investigators"] = "; ".join(n for n in names if n) or None
    return md


def from_lipd(filename):
    """Read a LiPD ``.lpd`` file into dplPy-native objects.

    Returns a dict with:

    - ``'rwl'`` : the raw ring widths as a year-indexed DataFrame (series as
      columns), ready for ``dpl.detrend`` / ``dpl.chron``, or None.
    - ``'chronology'`` : the primary (headline) chronology as a year-indexed
      DataFrame (``trsgi`` plus any ``count``/``RBAR``/``EPS``), or None.
    - ``'chronologies'`` : a dict of *all* chronologies found (keyed by a label
      such as ``'standard'``/``'residual'``/``'arstan'``), from measurement tables
      and/or model summaries.
    - ``'metadata'`` : a ``dpl.metadata()``-shape dict from the ``geo`` block.
    - ``'dsname'`` : the LiPD dataset name.

    Requires the optional ``pylipd`` dependency: ``pip install "dplpy[lipd]"``.
    """
    require_pylipd()
    from pylipd import LiPD
    lpd = LiPD()
    lpd.load(str(filename))
    datasets = lpd.get_datasets()
    if not datasets:
        raise ValueError("no datasets found in " + str(filename))
    ds = datasets[0]

    rwl = None
    chronologies = {}
    headline = None
    for tbl in _all_tables(ds):
        variables = tbl.getVariables() or []
        yvar = _year_var(variables)
        if yvar is None:
            continue
        years = pd.Index([int(float(y)) for y in _values(yvar)], name="Year")
        rw_vars = [v for v in variables if str(v.getName()) == "ringWidth"]
        trsgi_vars = [v for v in variables if str(v.getName()) == "trsgi"]

        if trsgi_vars:                                     # a chronology table
            data = {}
            for v in sorted(variables, key=lambda v: v.getColumnNumber() or 0):
                if v is yvar:
                    continue
                data[str(v.getName())] = _values(v)
            frame = pd.DataFrame(data, index=years)
            label = _chron_label(trsgi_vars[0].getDescription(),
                                 "chron" + str(len(chronologies) + 1))
            key, k = label, 2
            while key in chronologies:
                key = label + str(k); k += 1
            chronologies[key] = frame
            if headline is None or trsgi_vars[0].isPrimary():
                headline = frame
        elif len(rw_vars) >= 2 and rwl is None:            # a raw-width table
            data = {}
            for v in sorted(rw_vars, key=lambda v: v.getColumnNumber() or 0):
                m = re.search(r"series\s+(\S+)", v.getDescription() or "")
                col = (m.group(1) if m else (v.getVariableId()
                       or "series" + str(v.getColumnNumber())))
                data[col] = _values(v)
            rwl = pd.DataFrame(data, index=years)

    meta = _read_metadata(ds)
    span = None
    for frame in ([rwl] if rwl is not None else []) + list(chronologies.values()):
        if frame is not None and len(frame.index):
            span = frame.index if span is None else span
    if span is not None:
        meta["first_year"], meta["last_year"] = int(min(span)), int(max(span))

    return {"dsname": ds.getName(), "rwl": rwl, "chronology": headline,
            "chronologies": chronologies, "metadata": meta}


def lipd_metadata(filename):
    """Read just the site metadata from a LiPD ``.lpd`` file, as a
    ``dpl.metadata()``-shape dict. Requires the optional ``pylipd`` dependency."""
    require_pylipd()
    from pylipd import LiPD
    lpd = LiPD()
    lpd.load(str(filename))
    datasets = lpd.get_datasets()
    if not datasets:
        raise ValueError("no datasets found in " + str(filename))
    return _read_metadata(datasets[0])
