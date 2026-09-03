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
# Description: LiPD (Linked Paleo Data) export for dplPy. Phase 1: write a
#   chronology (and, when available, the raw ring widths it was built from) to a
#   LiPD `.lpd` container, delegating all of the container/JSON-LD/BagIt machinery
#   to `pylipd` (an optional dependency: `pip install "dplpy[lipd]"`). dplPy just
#   maps its outputs onto pylipd's object model.
#
#   Design (see dev/lipd_integration_proposal.md, decision 10.4): the CHRONOLOGY
#   is written as the *primary* paleoData measurement table, so the standard
#   pylipd/Pyleoclim extraction (`get_timeseries`) returns it directly. The raw
#   ring widths, when supplied, go in a second measurement table. (pylipd's
#   `get_timeseries` normalises the chronology's `trsgi` to the LiPD ontology
#   umbrella name `ringWidth`; the stored variableName is `trsgi` and
#   `get_all_variable_names` reports `trsgi` -- the chronology is the primary
#   series either way.)
#
# example usage:
#   >>> import dplpy as dpl
#   >>> rwl  = dpl.readers("ca533.rwl")
#   >>> rwi  = dpl.detrend(rwl, fit="Spline", method="ratio")
#   >>> crn  = dpl.chron(rwi)
#   >>> dpl.to_lipd(crn, "ca533.lpd", rwl=rwl)     # metadata auto-filled from rwl

import os
import re

import pandas as pd

from ._lipd_support import require_pylipd
from .site_metadata import SiteMetadata
from . import lipd_vocab


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
    ontology's umbrella term -- which for ``trsgi`` is ``ringWidth``, making the
    chronology indistinguishable from raw ring widths in the file. Omitting it
    keeps our specific, correct term (``trsgi``) as the stored ``variableName``
    (and what ``get_all_variable_names`` reports); ``get_timeseries`` still
    normalises to the ontology name at read time. The 'past' long-name parts are
    not part of the LiPD Variable schema and are not emitted here."""
    spec = lipd_vocab.lipd_variable(varkey)
    d = {"variableName": spec["variableName"], "units": spec["units"],
         "number": number, "TSid": tsid}
    if extra:
        d.update(extra)
    return d


def to_lipd(chron, filename, rwl=None, metadata=None, dsname=None,
            column="std", archive_type=None):
    """Write a dplPy chronology to a LiPD ``.lpd`` file.

    Parameters
    ----------
    chron : pandas.DataFrame
        A chronology from ``dpl.chron()`` -- year-indexed, with the index column
        ``column`` (default ``'std'``) and, if present, ``'samp_depth'`` (written
        as the ``count`` / ``sampleCount`` companion). Written as the **primary**
        paleoData measurement table.
    filename : str
        Output path. ``.lpd`` is appended if missing.
    rwl : pandas.DataFrame, optional
        The raw ring widths the chronology was built from (e.g. from
        ``dpl.readers()``). When given, written as a second measurement table, and
        -- if ``metadata`` is None -- used to auto-fill site metadata from its
        ``dplpy_metadata`` attrs.
    metadata : SiteMetadata or dict, optional
        Site metadata for the ``geo`` block. A ``SiteMetadata`` (see
        ``dpl.SiteMetadata``) or a metadata dict. Defaults to what ``rwl`` carries,
        else empty.
    dsname : str, optional
        LiPD dataset name. Defaults to the site id / site name / filename stem.
    column : str, default 'std'
        Which column of ``chron`` is the chronology index to write as ``trsgi``.
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

    if not isinstance(chron, pd.DataFrame):
        raise TypeError("chron must be a pandas DataFrame from dpl.chron(), not "
                        + str(type(chron)))
    if column not in chron.columns:
        raise ValueError("chronology column '" + str(column) + "' not found; "
                         "available columns: " + str(list(chron.columns)))
    if rwl is not None and not isinstance(rwl, pd.DataFrame):
        raise TypeError("rwl must be a pandas DataFrame of ring widths or None.")

    # -- site metadata (auto from rwl if not given) ------------------------- #
    if metadata is not None:
        meta = SiteMetadata.coerce(metadata)
    elif isinstance(rwl, pd.DataFrame) and rwl.attrs.get("dplpy_metadata"):
        meta = SiteMetadata.from_rwl(rwl)
    else:
        meta = SiteMetadata()

    if not str(filename).lower().endswith(".lpd"):
        filename = str(filename) + ".lpd"
    # Dataset name: default to the (meaningful) filename stem the caller chose,
    # overridable via `dsname`. A purely-numeric name (e.g. the ITRDB id "712")
    # must be avoided -- pylipd coerces it to a float ("712.0") and can no longer
    # resolve the dataset (get_timeseries returns nothing), so we guard against it.
    file_stem = os.path.splitext(os.path.basename(filename))[0]
    dsname = str(dsname or file_stem or "dataset").strip()
    if re.fullmatch(r"[0-9.]+", dsname) or not dsname:
        dsname = "dplpy-" + (dsname.replace(".", "-") or "dataset")
    stem = _sanitize(dsname)

    # -- PRIMARY measurement table = the chronology ------------------------- #
    cdf = pd.DataFrame({"year": [int(y) for y in chron.index],
                        "trsgi": chron[column].to_numpy(dtype=float)})
    cattrs = {
        "year": _attrs("year", 1, _tsid(stem, "year")),
        "trsgi": _attrs("trsgi", 2, _tsid(stem, "trsgi"),
                        {"isPrimary": True,
                         "description": "Standard chronology built with dplPy "
                                        "(column '" + str(column) + "')"}),
    }
    if "samp_depth" in chron.columns:
        cdf["count"] = chron["samp_depth"].to_numpy(dtype=float)
        cattrs["count"] = _attrs("sample_count", 3, _tsid(stem, "sampleCount"))
    cdf.attrs = cattrs
    primary = DataTable()
    primary.setFileName(stem + ".paleo1measurement1.csv")
    primary.setMissingValue("NaN")
    primary.setDataFrame(cdf)
    tables = [primary]

    # -- SECOND measurement table = raw ring widths (optional) -------------- #
    if isinstance(rwl, pd.DataFrame):
        series = [str(c) for c in rwl.columns]
        mdf = rwl.reset_index()
        mdf.columns = ["year", *series]
        mattrs = {"year": _attrs("year", 1, _tsid(stem, "rw", "year"))}
        for i, s in enumerate(series, start=2):
            mattrs[s] = _attrs("ring_width", i, _tsid(stem, s, "ringWidth"),
                               {"description": "Raw ring width measurements from "
                                               "series " + s,
                                "longName": s + " ring width"})
        mdf.attrs = mattrs
        widths = DataTable()
        widths.setFileName(stem + ".paleo1measurement2.csv")
        widths.setMissingValue("NaN")
        widths.setDataFrame(mdf)
        tables.append(widths)

    # -- assemble the Dataset ---------------------------------------------- #
    pdo = PaleoData()
    pdo.setName("paleo1")
    pdo.setMeasurementTables(tables)

    ds = Dataset()
    ds.setName(dsname)
    ds.setPaleoData([pdo])
    ds.set_non_standard_property("archiveType", archive_type or lipd_vocab.ARCHIVE_TYPE)

    loc = Location()
    _set_if = lambda setter, value: setter(str(value)) if value is not None else None
    _set_if(loc.setLatitude, meta.latitude)
    _set_if(loc.setLongitude, meta.longitude)
    _set_if(loc.setElevation, meta.elevation_m)
    _set_if(loc.setSiteName, meta.site_name)
    _set_if(loc.setLocationName, meta.country_region)
    ds.setLocation(loc)

    lpd = LiPD()
    lpd.load_datasets([ds])
    lpd.create_lipd(dsname, filename)
    return filename
