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

# Title: tridas.py
# Description: Read and write TRiDaS (Tree Ring Data Standard) XML files.
#   dpl.read_tridas() parses a TRiDaS file into a typed TridasProject container
#   (measurement tables, derived chronologies, undated/floating series, and the
#   project -> object -> element -> sample -> radius -> series metadata). A
#   standalone dpl.write_tridas() emits a TRiDaS 1.2.2 file from a ring-width
#   frame and/or a chronology frame plus a TridasMetadata.
#
#   dpl.readers() sniffs a <tridas> element and, for the common single-table
#   file, returns a plain ring-width DataFrame; a multi-table or chronology file
#   is sent here with a clear message.
#
#   Design decisions (see dev/tridas_design_2026-10-14.md):
#   - Missing/absent values are NaN, never 0 (dplR uses 0 internally); a measured
#     0 is kept as a real locally-absent ring.
#   - Units convert to millimetres and reuse dplPy's precision model
#     (df.attrs["dplpy_precision"]).
#   - Years round-trip through TRiDaS firstYear/lastYear AD/BC suffixes; dplPy's
#     internal year axis is a continuous integer index (astronomical numbering,
#     year 0 == 1 BC).
#   - Not a port of dplR's SAX state machine: a stdlib ElementTree tree walk.
#   - v1 keeps duplicate series IDs separate with a warning (no series merging).

import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# The namespace dplR writes (TRiDaS 1.2.2). Reading is namespace-agnostic (we
# match on local names), so 1.1/1.2.x files parse too; writing emits 1.2.2.
TRIDAS_NS = "http://www.tridas.org/1.2.2"

# A TRiDaS unit name -> the width in millimetres of one integer step in that
# unit. Reading multiplies raw values by this; it is also the dplPy precision the
# unit implies, so the same table feeds df.attrs["dplpy_precision"].
_UNIT_TO_MM = {
    "micrometres": 0.001,
    "1/100th millimetres": 0.01,
    "1/50th millimetres": 0.02,
    "1/20th millimetres": 0.05,
    "1/10th millimetres": 0.1,
    "millimetres": 1.0,
    "centimetres": 10.0,
    "metres": 1000.0,
}

# A dplPy precision (mm per step) -> (TRiDaS unit, integer multiplier) for
# writing. value_integer = round(value_mm * multiplier).
_PREC_TO_UNIT = {
    0.001: ("micrometres", 1000),
    0.01: ("1/100th millimetres", 100),
    0.02: ("1/50th millimetres", 50),
    0.05: ("1/20th millimetres", 20),
    0.1: ("1/10th millimetres", 10),
    1.0: ("millimetres", 1),
    10.0: ("centimetres", 0.1),
    1000.0: ("metres", 0.001),
}

_RING_WIDTH = "ring width"


# --------------------------------------------------------------------------- #
# Typed container returned by read_tridas()
# --------------------------------------------------------------------------- #
@dataclass
class TridasMetadata:
    """The project/object/element/sample/radius/series metadata of a TRiDaS file.

    Project-level fields are named (they are the mandatory ones in Brewer's list
    and what a .crn-style export needs). The object hierarchy is kept as nested
    dicts under :attr:`objects` so the whole tree stays navigable without a deep
    class hierarchy. :attr:`laboratories` and :attr:`extra` capture the rest.
    """

    project_title: Optional[str] = None
    project_types: list = field(default_factory=list)
    category: Optional[str] = None
    investigator: Optional[str] = None
    period: Optional[str] = None
    description: Optional[str] = None
    laboratories: list = field(default_factory=list)   # list of dicts
    objects: list = field(default_factory=list)         # nested dict hierarchy
    extra: dict = field(default_factory=dict)


@dataclass
class TridasProject:
    """Parsed TRiDaS file.

    Attributes
    ----------
    measurements : dict of str -> DataFrame
        Dated measurement series grouped into ring-width tables, keyed by
        ``"object | variable | unit | taxon"``. Each frame is Year-indexed with
        series in columns (the dplPy rwl shape) and carries ``df.attrs["kind"] =
        "rwl"``.
    derived : dict of str -> DataFrame
        Chronologies (TRiDaS derivedSeries), keyed by series title. Each frame is
        Year-indexed with the chronology column and, when the file records per
        value counts, a ``samp_depth`` column; ``df.attrs["kind"] = "crn"``.
    undated : dict of str -> Series
        Measurement series with no calendar years (floating series), keyed by
        series id. Position-indexed; feed these to dpl.xdate_floater().
    metadata : TridasMetadata
        The full metadata hierarchy.
    """

    measurements: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    undated: dict = field(default_factory=dict)
    metadata: TridasMetadata = field(default_factory=TridasMetadata)

    def rwl(self) -> pd.DataFrame:
        """Return the single ring-width table, or raise if the file holds more
        than one measurement table (then use :attr:`measurements`)."""
        if len(self.measurements) == 1:
            return next(iter(self.measurements.values()))
        if len(self.measurements) == 0:
            raise ValueError("this TRiDaS file has no dated measurement series; "
                             "see .derived (chronologies) or .undated.")
        raise ValueError(
            "this TRiDaS file has %d measurement tables; pick one from "
            ".measurements (keys: %s)." % (len(self.measurements),
                                           ", ".join(self.measurements))
        )


# --------------------------------------------------------------------------- #
# Small XML helpers (namespace-agnostic: match on local names)
# --------------------------------------------------------------------------- #
def _local(tag):
    """Local name of an ElementTree tag, dropping any ``{namespace}`` prefix."""
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _text(el):
    if el is None or el.text is None:
        return None
    t = el.text.strip()
    return t if t else None


def _decode_year(value, suffix):
    """TRiDaS firstYear/lastYear (a positive int + AD/BC suffix) -> a dplPy
    integer year on the continuous astronomical axis (year 0 == 1 BC)."""
    if value is None:
        return None
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if suffix is not None and str(suffix).upper() == "BC":
        return 1 - n          # 1 BC -> 0, 2 BC -> -1, ...
    return n


def _encode_year(year):
    """A dplPy integer year -> (positive value, "AD"/"BC") for TRiDaS."""
    if year < 1:
        return (1 - year, "BC")
    return (year, "AD")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_tridas(filename, warn_units=True):
    """Read a TRiDaS (Tree Ring Data Standard) XML file.

    Parameters
    ----------
    filename : str
        path to a ``.xml`` TRiDaS file.
    warn_units : bool, default True
        warn when a measurement series uses a variable that is not a width in a
        length unit (e.g. a density), which dplPy does not convert to mm. Such
        series are kept out of the ring-width tables.

    Returns
    -------
    TridasProject
        A typed container: ``.measurements`` (ring-width tables), ``.derived``
        (chronologies), ``.undated`` (floating series), ``.metadata``. For the
        common single-table file, ``dpl.readers()`` returns the ring-width
        DataFrame directly instead.

    Notes
    -----
    Missing values become NaN (not 0); units are converted to millimetres;
    years use the dplPy continuous integer axis. See
    ``dev/tridas_design_2026-10-14.md`` for the full design.

    References
    ----------
    .. [1] Jansma, Brewer & Zandhuis (2010), Dendrochronologia 28, 99-130.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    tree = ET.parse(filename)
    root = tree.getroot()
    if _local(root.tag) != "tridas":
        # some files wrap projects directly; accept a <project> root too
        if _local(root.tag) != "project":
            raise ValueError(
                os.path.basename(filename) + " is not a TRiDaS file "
                "(root element is <" + str(_local(root.tag)) + ">, expected "
                "<tridas>).")

    projects = [root] if _local(root.tag) == "project" else _children(root, "project")
    if not projects:
        raise ValueError("no <project> element found in " + os.path.basename(filename))

    # v1 reads the first project (TRiDaS allows several; multi-project files are
    # rare and a later refinement). Record if we skipped any.
    if len(projects) > 1:
        warnings.warn("read_tridas: %d projects in the file; reading the first "
                      "only (multi-project support is a later addition)."
                      % len(projects))
    proj = projects[0]

    meta = _read_project_metadata(proj)

    # Accumulators keyed by group string -> {colname: pd.Series(year->mm)}
    measurement_groups = {}
    measurement_prec = {}       # group -> {colname: precision mm}
    undated = {}
    derived = {}

    for obj in _children(proj, "object"):
        obj_dict = _read_object(obj, meta, measurement_groups, measurement_prec,
                                undated, warn_units)
        meta.objects.append(obj_dict)

    for ds in _children(proj, "derivedSeries"):
        name, frame = _read_derived(ds, warn_units)
        if frame is not None:
            name = _unique_key(name, derived)
            derived[name] = frame

    # Materialize measurement groups into Year-indexed frames.
    measurements = {}
    for key, cols in measurement_groups.items():
        frame = _assemble_group(cols, measurement_prec.get(key, {}))
        measurements[key] = frame

    return TridasProject(measurements=measurements, derived=derived,
                         undated=undated, metadata=meta)


def _read_project_metadata(proj):
    meta = TridasMetadata()
    meta.project_title = _text(_child(proj, "title"))
    meta.project_types = [t for t in (_text(e) for e in _children(proj, "type")) if t]
    meta.category = _text(_child(proj, "category"))
    meta.investigator = _text(_child(proj, "investigator"))
    meta.period = _text(_child(proj, "period"))
    meta.description = _text(_child(proj, "description"))
    for lab in _children(proj, "laboratory"):
        name_el = _child(lab, "name")
        meta.laboratories.append({
            "name": _text(name_el),
            "acronym": name_el.get("acronym") if name_el is not None else None,
        })
    return meta


def _read_object(obj, meta, groups, prec, undated, warn_units, parent_titles=None):
    """Walk one <object> (and its sub-objects) collecting series into groups."""
    obj_title = _text(_child(obj, "title"))
    titles = (parent_titles or []) + [obj_title or ""]
    obj_path = " / ".join([t for t in titles if t])
    obj_dict = {
        "title": obj_title,
        "type": _text(_child(obj, "type")),
        "elements": [],
        "objects": [],
    }
    for sub in _children(obj, "object"):
        obj_dict["objects"].append(
            _read_object(sub, meta, groups, prec, undated, warn_units, titles))
    for elem in _children(obj, "element"):
        obj_dict["elements"].append(
            _read_element(elem, obj_path, groups, prec, undated, warn_units))
    return obj_dict


def _read_element(elem, obj_path, groups, prec, undated, warn_units):
    taxon = _text(_child(elem, "taxon"))
    el_dict = {
        "title": _text(_child(elem, "title")),
        "taxon": taxon,
        "samples": [],
    }
    for samp in _children(elem, "sample"):
        samp_dict = {
            "title": _text(_child(samp, "title")),
            "type": _text(_child(samp, "type")),
            "radii": [],
        }
        for rad in _children(samp, "radius"):
            rad_dict = {"title": _text(_child(rad, "title")), "series": []}
            for ms in _children(rad, "measurementSeries"):
                info = _read_measurement_series(
                    ms, obj_path, taxon, groups, prec, undated, warn_units)
                rad_dict["series"].append(info)
            samp_dict["radii"].append(rad_dict)
        el_dict["samples"].append(samp_dict)
    return el_dict


def _read_measurement_series(ms, obj_path, taxon, groups, prec, undated, warn_units):
    title = _text(_child(ms, "title"))
    comments = _text(_child(ms, "comments"))   # dplPy stores the column name here
    colname = comments or title or "series"

    first_year, last_year = _read_interpretation(ms)
    values_el = _child(ms, "values")
    variable, unit, raw = _read_values(values_el)

    info = {
        "id": colname, "title": title, "variable": variable, "unit": unit,
        "first_year": first_year, "last_year": last_year,
        "measuring_method": _read_measuring_method(ms),
    }

    if not raw:
        return info

    # Non-width variables (densities etc.) are not ring widths in mm; keep them
    # out of the ring-width tables but still report them in metadata.
    is_width = variable is None or variable == _RING_WIDTH or "width" in (variable or "")
    factor = _UNIT_TO_MM.get(unit)
    if not is_width or factor is None:
        if warn_units:
            warnings.warn("read_tridas: series %r has variable %r / unit %r that "
                          "is not a length width; left out of the ring-width "
                          "tables." % (colname, variable, unit))
        return info

    values_mm = [np.nan if v is None else v * factor for v in raw]

    if first_year is None:
        # Undated / floating series: no calendar axis.
        key = _unique_key(colname, undated)
        undated[key] = pd.Series(values_mm, name=key, dtype=float)
        return info

    years = list(range(first_year, first_year + len(values_mm)))
    s = pd.Series(values_mm, index=pd.Index(years, name="Year"), dtype=float)

    group_key = "%s | %s | %s | %s" % (obj_path or "", variable or _RING_WIDTH,
                                       unit or "", taxon or "")
    cols = groups.setdefault(group_key, {})
    pcols = prec.setdefault(group_key, {})
    ucol = _unique_key(colname, cols)
    cols[ucol] = s
    pcols[ucol] = _UNIT_TO_MM.get(unit, 1.0)
    return info


def _read_interpretation(el):
    interp = _child(el, "interpretation")
    if interp is None:
        return (None, None)
    fy = _child(interp, "firstYear")
    ly = _child(interp, "lastYear")
    first = _decode_year(_text(fy), fy.get("suffix") if fy is not None else None)
    last = _decode_year(_text(ly), ly.get("suffix") if ly is not None else None)
    return (first, last)


def _read_values(values_el):
    """Return (variable, unit, [float|None, ...]) from a <values> element."""
    if values_el is None:
        return (None, None, [])
    var_el = _child(values_el, "variable")
    variable = None
    if var_el is not None:
        variable = var_el.get("normalTridas") or var_el.get("normal") or _text(var_el)
    unit_el = _child(values_el, "unit")
    unit = None
    if unit_el is not None:
        unit = unit_el.get("normalTridas") or unit_el.get("normal") or _text(unit_el)
    raw = []
    for v in _children(values_el, "value"):
        raw.append(_value_number(v))
    return (variable, unit, raw)


def _value_number(v):
    """A <value> -> float, or None for a missing/non-numeric value (-> NaN).
    A real 0 is kept as 0.0 (a locally absent ring), never turned into NaN."""
    attr = v.get("value")
    if attr is None:
        return None
    try:
        return float(attr)
    except (TypeError, ValueError):
        return None


def _read_measuring_method(ms):
    mm = _child(ms, "measuringMethod")
    if mm is None:
        return None
    return mm.get("normalTridas") or mm.get("normal") or _text(mm)


def _read_derived(ds, warn_units):
    """Read a <derivedSeries> into (title, Year-indexed chronology frame)."""
    title = _text(_child(ds, "title")) or _text(_child(ds, "comments")) or "derived"
    first_year, _ = _read_interpretation(ds)
    values_el = _child(ds, "values")
    if values_el is None:
        return (title, None)

    var_el = _child(values_el, "variable")
    variable = None
    if var_el is not None:
        variable = var_el.get("normalTridas") or var_el.get("normal") or _text(var_el)
    unit_el = _child(values_el, "unit")
    unitless = _child(values_el, "unitless") is not None
    unit = None
    if unit_el is not None:
        unit = unit_el.get("normalTridas") or unit_el.get("normal") or _text(unit_el)
    factor = 1.0 if (unitless or unit is None) else _UNIT_TO_MM.get(unit, 1.0)

    vals, counts = [], []
    have_counts = False
    for v in _children(values_el, "value"):
        num = _value_number(v)
        vals.append(np.nan if num is None else num * factor)
        c = v.get("count")
        if c is not None:
            have_counts = True
        try:
            counts.append(float(c) if c is not None else np.nan)
        except (TypeError, ValueError):
            counts.append(np.nan)

    if not vals or first_year is None:
        # An undated chronology is unusual; store values with a positional index.
        idx = pd.RangeIndex(len(vals))
    else:
        idx = pd.Index(range(first_year, first_year + len(vals)), name="Year")

    data = {title: vals}
    if have_counts:
        data["samp_depth"] = counts
    frame = pd.DataFrame(data, index=idx)
    frame.attrs["kind"] = "crn"
    return (title, frame)


def _assemble_group(cols, prec):
    """Build a Year-indexed ring-width frame from {colname: Series(year->mm)}."""
    frame = pd.concat(cols.values(), axis=1)
    frame.columns = list(cols.keys())
    frame = frame.sort_index()
    frame.index.name = "Year"
    frame.attrs["kind"] = "rwl"
    if prec:
        frame.attrs["dplpy_precision"] = dict(prec)
        frame.attrs["dplpy_mixed_precision"] = (len(set(prec.values())) > 1)
    return frame


def _unique_key(name, existing):
    """Return `name`, or `name_2`, `name_3`, ... if already present. Warns once
    per collision (v1 keeps duplicate series separate rather than merging)."""
    if name not in existing:
        return name
    i = 2
    while ("%s_%d" % (name, i)) in existing:
        i += 1
    newname = "%s_%d" % (name, i)
    warnings.warn("read_tridas: duplicate series id %r; kept separately as %r "
                  "(v1 does not merge series)." % (name, newname))
    return newname


# --------------------------------------------------------------------------- #
# readers() delegation: the single-table contract
# --------------------------------------------------------------------------- #
def read_tridas_single(filename, warn_units=True):
    """Return one ring-width DataFrame for the common single-table TRiDaS file,
    or raise pointing to read_tridas() when the file is richer. Used by
    dpl.readers() so a plain ``dpl.readers("x.xml")`` behaves like any read."""
    proj = read_tridas(filename, warn_units=warn_units)
    if proj.derived:
        raise ValueError(
            os.path.basename(filename) + " contains derived chronologies as well "
            "as (or instead of) raw measurements; use dpl.read_tridas() to get "
            "the full TridasProject.")
    if len(proj.measurements) != 1:
        raise ValueError(
            os.path.basename(filename) + " contains %d measurement tables; "
            "dpl.readers() returns a single table only. Use dpl.read_tridas() "
            "for the full TridasProject (keys: %s)."
            % (len(proj.measurements), ", ".join(proj.measurements)))
    frame = next(iter(proj.measurements.values()))
    # Attach light site metadata for interoperability with SiteMetadata.
    md = proj.metadata
    site = _first_object(md)
    frame.attrs["dplpy_metadata"] = {
        "site_name": (site or {}).get("title"),
        "species_name": _first_taxon(site),
        "investigators": md.investigator,
    }
    frame.attrs["dplpy_tridas_metadata"] = md
    return frame


def _first_object(md):
    return md.objects[0] if md.objects else None


def _first_taxon(site):
    if not site:
        return None
    for el in site.get("elements", []):
        if el.get("taxon"):
            return el["taxon"]
    return None


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_tridas(fname, rwl=None, crn=None, metadata=None, prec=None,
                 ids=None, taxon=None, variable=_RING_WIDTH,
                 measuring_method="unknown", validate=True):
    """Write a TRiDaS 1.2.2 file from ring widths and/or a chronology.

    Parameters
    ----------
    fname : str
        output path. ``.xml`` is appended if no extension is given.
    rwl : pandas DataFrame, optional
        ring widths (Year-indexed, series in columns), as from dpl.readers().
        Written as measurementSeries under one object (site).
    crn : pandas DataFrame, optional
        a chronology from dpl.chron(); columns named ``samp_depth`` carry sample
        depth and pair with the preceding chronology column. Written as
        derivedSeries, unitless, with sample depth as each value's ``count``.
    metadata : TridasMetadata or dict, optional
        project/site metadata. Mandatory fields missing with ``validate=True``
        raise a ValueError naming them (``type``/``investigator`` are filled with
        "unknown" where the standard allows).
    prec : float, optional
        measurement precision in mm (0.001, 0.01, 0.02, 0.05, 0.1, 1, 10, 1000).
        ``None`` (default) writes millimetres without rounding.
    ids : pandas DataFrame, optional
        tree/core[/radius/measurement] grouping, as from dpl.read_ids(). Default:
        each series is its own element (tree).
    taxon : str, optional
        taxon for the elements. Mandatory per the standard when writing rwl.
    variable : str, default "ring width"
        the measured variable (controlled vocabulary).
    measuring_method : str, default "unknown"
        free-text measuring method (mandatory per the standard).
    validate : bool, default True
        enforce the mandatory-field set.

    Returns
    -------
    str
        the path written.
    """
    if rwl is None and crn is None:
        raise ValueError("write_tridas: nothing to write -- pass rwl and/or crn.")
    if not fname.lower().endswith(".xml"):
        fname = fname + ".xml"

    md = _coerce_metadata(metadata)
    if validate:
        _validate_mandatory(md, rwl, crn, taxon, variable)

    ET.register_namespace("", TRIDAS_NS)
    root = _el("tridas")
    proj = _sub(root, "project")
    _leaf(proj, "title", md.project_title or "")
    for t in (md.project_types or ["unknown"]):
        _leaf(proj, "type", t)
    if md.laboratories:
        for lab in md.laboratories:
            lab_el = _sub(proj, "laboratory")
            nm = _leaf(lab_el, "name", lab.get("name") or "")
            if lab.get("acronym"):
                nm.set("acronym", lab["acronym"])
    _leaf(proj, "category", md.category or "")
    _leaf(proj, "investigator", md.investigator or "unknown")
    _leaf(proj, "period", md.period or "")

    if rwl is not None:
        _write_object(proj, rwl, md, prec, ids, taxon, variable, measuring_method)
    if crn is not None:
        _write_derived(proj, crn, variable)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")       # Python >= 3.9
    except AttributeError:
        pass
    tree.write(fname, xml_declaration=True, encoding="utf-8")
    return fname


def _write_object(proj, rwl, md, prec, ids, taxon, variable, measuring_method):
    site = _first_object(md) or {}
    obj = _sub(proj, "object")
    _leaf(obj, "title", site.get("title") or "")
    _leaf(obj, "type", site.get("type") or "unknown")

    cnames = [str(c) for c in rwl.columns]
    data_unit, mult = _resolve_prec(prec)
    years = [int(y) for y in rwl.index]

    grouping = _resolve_ids(ids, cnames)
    for tree_id, cols in grouping:
        elem = _sub(obj, "element")
        _leaf(elem, "title", str(tree_id))
        _leaf(elem, "taxon", taxon or "")
        # v1: one sample/radius per element, one measurementSeries per column.
        samp = _sub(elem, "sample")
        _leaf(samp, "title", str(tree_id))
        _leaf(samp, "type", "core")
        rad = _sub(samp, "radius")
        _leaf(rad, "title", "1")
        for col in cols:
            _write_measurement_series(rad, rwl[col], col, years, data_unit, mult,
                                      variable, measuring_method)


def _write_measurement_series(rad, series, colname, years, data_unit, mult,
                              variable, measuring_method):
    ms = _sub(rad, "measurementSeries")
    _leaf(ms, "title", str(colname))
    _leaf(ms, "comments", str(colname))          # dplPy column name round-trips here
    _leaf(ms, "measuringMethod", measuring_method or "unknown")

    vals = series.to_numpy(dtype=float)
    present = ~np.isnan(vals)
    if not present.any():
        # No data: write an empty measured span so the file stays well formed.
        interp = _sub(ms, "interpretation")
        values = _sub(ms, "values")
        _write_variable_unit(values, variable, data_unit)
        return
    first_i = int(np.argmax(present))
    last_i = len(present) - 1 - int(np.argmax(present[::-1]))
    first_year = years[first_i]
    last_year = years[last_i]

    interp = _sub(ms, "interpretation")
    _write_year(interp, "firstYear", first_year)
    _write_year(interp, "lastYear", last_year)

    values = _sub(ms, "values")
    _write_variable_unit(values, variable, data_unit)
    for i in range(first_i, last_i + 1):
        v = vals[i]
        ve = _sub(values, "value")
        if not np.isnan(v):
            ve.set("value", _fmt_value(v, mult))
        # an interior NaN is written as an empty <value/> -> reads back as NaN


def _write_derived(proj, crn, variable):
    if isinstance(crn, pd.DataFrame):
        frames = [crn]
    else:
        frames = list(crn)
    for frame in frames:
        cols = [str(c) for c in frame.columns]
        depth_cols = [c for c in cols if c.lower().replace(".", "_").startswith("samp_depth")
                      or c.lower().startswith("samp.depth")]
        series_cols = [c for c in cols if c not in depth_cols]
        years = [int(y) for y in frame.index]
        for j, col in enumerate(series_cols):
            depth = frame[depth_cols[min(j, len(depth_cols) - 1)]] if depth_cols else None
            _write_one_derived(proj, frame[col], col, years, depth, variable)


def _write_one_derived(proj, series, colname, years, depth, variable):
    ds = _sub(proj, "derivedSeries")
    _leaf(ds, "title", str(colname))
    _leaf(ds, "type", "")
    _sub(ds, "linkSeries")                   # dplR writes this empty; we match for now
    vals = series.to_numpy(dtype=float)
    present = ~np.isnan(vals)
    interp = _sub(ds, "interpretation")
    if present.any():
        first_i = int(np.argmax(present))
        last_i = len(present) - 1 - int(np.argmax(present[::-1]))
        _write_year(interp, "firstYear", years[first_i])
        _write_year(interp, "lastYear", years[last_i])
    else:
        first_i, last_i = 0, -1
    values = _sub(ds, "values")
    _leaf(values, "variable", "").set("normalTridas", variable or _RING_WIDTH)
    _sub(values, "unitless")                 # chronologies are indices: unitless
    depth_vals = depth.to_numpy(dtype=float) if depth is not None else None
    for i in range(first_i, last_i + 1):
        ve = _sub(values, "value")
        if not np.isnan(vals[i]):
            ve.set("value", repr(float(vals[i])))
        if depth_vals is not None and not np.isnan(depth_vals[i]):
            ve.set("count", str(int(round(depth_vals[i]))))


# -- write helpers ---------------------------------------------------------- #
def _el(name):
    return ET.Element("{%s}%s" % (TRIDAS_NS, name))


def _sub(parent, name):
    return ET.SubElement(parent, "{%s}%s" % (TRIDAS_NS, name))


def _leaf(parent, name, text):
    e = _sub(parent, name)
    if text is not None and text != "":
        e.text = str(text)
    return e


def _write_year(parent, name, year):
    value, suffix = _encode_year(int(year))
    e = _sub(parent, name)
    e.text = str(value)
    e.set("suffix", suffix)
    return e


def _write_variable_unit(values, variable, data_unit):
    _leaf(values, "variable", "").set("normalTridas", variable or _RING_WIDTH)
    _leaf(values, "unit", "").set("normalTridas", data_unit)


def _fmt_value(v, mult):
    if mult == 1:
        # millimetres without rounding
        return repr(float(v))
    return str(int(round(v * mult)))


def _resolve_prec(prec):
    if prec is None:
        return ("millimetres", 1)
    if prec not in _PREC_TO_UNIT:
        warnings.warn("write_tridas: unknown prec %r; writing millimetres "
                      "unrounded." % (prec,))
        return ("millimetres", 1)
    return _PREC_TO_UNIT[prec]


def _resolve_ids(ids, cnames):
    """Return a list of (tree_id, [colnames]) groups. Default: each column its
    own element. With an `ids` frame (dpl.read_ids output), group by tree."""
    if ids is None:
        return [(c, [c]) for c in cnames]
    groups = {}
    order = []
    for c in cnames:
        if c in ids.index:
            tree = ids.loc[c, "tree"]
        else:
            tree = c
        if tree not in groups:
            groups[tree] = []
            order.append(tree)
        groups[tree].append(c)
    return [(t, groups[t]) for t in order]


def _coerce_metadata(metadata):
    if metadata is None:
        return TridasMetadata()
    if isinstance(metadata, TridasMetadata):
        return metadata
    if isinstance(metadata, dict):
        md = TridasMetadata()
        for k, v in metadata.items():
            if hasattr(md, k):
                setattr(md, k, v)
            else:
                md.extra[k] = v
        return md
    raise TypeError("metadata must be a TridasMetadata or dict, not "
                    + str(type(metadata)))


def _validate_mandatory(md, rwl, crn, taxon, variable):
    """Raise ValueError naming missing mandatory fields. type and investigator
    are allowed to default to 'unknown' and are not required here."""
    missing = []
    if not md.project_title:
        missing.append("project.title")
    if not md.category:
        missing.append("project.category")
    if not md.period:
        missing.append("project.period")
    site = _first_object(md) or {}
    if rwl is not None:
        if not site.get("title"):
            missing.append("object.title")
        if not taxon:
            missing.append("element.taxon")
    if missing:
        raise ValueError(
            "write_tridas: missing mandatory TRiDaS field(s): "
            + ", ".join(missing)
            + ". Provide them in `metadata` (and `taxon=`), or pass "
            "validate=False to write with empty placeholders.")
