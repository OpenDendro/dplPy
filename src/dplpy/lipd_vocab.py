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

# Title: lipd_vocab.py
# Description: The one place that maps dplPy quantities to controlled vocabulary
#   for LiPD export. Two vocabularies are involved and both are served from the
#   same table:
#     (1) LiPD's own variableName / units terms (lipdverse.org/vocabulary), used
#         directly in the metadata.jsonld a LiPD file carries; and
#     (2) NOAA's PaST Thesaurus nine-part variable name (the WDS-Paleo standard),
#         which lipdverse links to and which we can emit as a long name.
#   This module has NO third-party dependencies (it is pure data + helpers), so
#   it lives in core dplPy; the actual LiPD writer (planned) reads from here so
#   the vocabulary lives in exactly one place and is easy to update if lipdverse
#   revises its terms.
#
# Confirmed against real lipdverse dendro data (the NAm2k compilation): a
# tree-ring archive is archiveType "Wood"; a standard chronology index is
# variableName "trsgi"; raw widths are "ringWidth" (mm); chronology companions
# are "sampleCount", "RBAR", "EPS".

# ---- archive / proxy (record-level) ---------------------------------------- #
ARCHIVE_TYPE = "Wood"        # LiPD archiveType for a tree-ring record
PROXY = "TRW"                # tree-ring width

# ---- the nine PaST fields, in order (see NOAA Variable_naming_guide) -------- #
# What (req), Material, Error, Units (req), Seasonality, Data Type (req),
# Detail, Method, Data Format (req; "N" numeric / "C" character).
PAST_FIELDS = ("what", "material", "error", "units", "seasonality",
               "data_type", "detail", "method", "data_format")

# ---- dplPy quantity  ->  LiPD variable spec (+ PaST parts) ------------------ #
# Keys are dplPy-side identifiers; each entry carries the LiPD variableName, its
# units, the LiPD hasStandardVariable, and the fixed PaST field values for that
# quantity (variable, per-series Detail/Method are filled in by the writer).
LIPD_VARIABLES = {
    "year": {
        "variableName": "year", "units": "yr AD",
        "hasStandardVariable": "year",
        "past": {"what": "year", "units": "year Common Era",
                 "data_type": "Tree Ring", "data_format": "N"},
    },
    "ring_width": {
        "variableName": "ringWidth", "units": "mm",
        "hasStandardVariable": "ringWidth",
        "past": {"what": "ring width", "units": "millimeter",
                 "data_type": "Tree Ring", "data_format": "N"},
    },
    "trsgi": {                       # the standard chronology index (dpl.chron std)
        "variableName": "trsgi", "units": "unitless",
        "hasStandardVariable": "trsgi",
        "past": {"what": "standardized growth index", "units": "dimensionless",
                 "data_type": "Tree Ring", "data_format": "N"},
    },
    "sample_count": {                # sample depth per year (dpl.chron samp_depth)
        "variableName": "sampleCount", "units": "count",
        "hasStandardVariable": "sampleCount",
        "past": {"what": "sample depth", "units": "number",
                 "data_type": "Tree Ring", "data_format": "N"},
    },
    "rbar": {                        # running mean interseries correlation
        "variableName": "RBAR", "units": "unitless",
        "hasStandardVariable": "RBAR",
        "past": {"what": "mean interseries correlation", "units": "dimensionless",
                 "data_type": "Tree Ring", "data_format": "N"},
    },
    "eps": {                         # expressed population signal
        "variableName": "EPS", "units": "unitless",
        "hasStandardVariable": "EPS",
        "past": {"what": "expressed population signal", "units": "dimensionless",
                 "data_type": "Tree Ring", "data_format": "N"},
    },
}


def lipd_variable(key: str) -> dict:
    """Return a *copy* of the LiPD variable spec for a dplPy quantity key
    (e.g. 'ring_width', 'trsgi'). Raises KeyError for an unknown key."""
    if key not in LIPD_VARIABLES:
        raise KeyError("unknown dplPy variable key '" + str(key) + "'; known: "
                       + ", ".join(sorted(LIPD_VARIABLES)))
    spec = dict(LIPD_VARIABLES[key])
    spec["past"] = dict(spec.get("past", {}))   # copy the nested dict too
    return spec


def past_long_name(what, units, data_type="Tree Ring", data_format="N",
                   material="", error="", seasonality="", detail="", method=""):
    """Build a NOAA PaST nine-part variable long name: the nine fields joined by
    ', ' in the fixed order (What, Material, Error, Units, Seasonality, Data
    Type, Detail, Method, Data Format). Empty fields are kept as empty slots, per
    the guide ("ensure there is a comma separating each category, even where no
    value exists")."""
    parts = {"what": what, "material": material, "error": error, "units": units,
             "seasonality": seasonality, "data_type": data_type,
             "detail": detail, "method": method, "data_format": data_format}
    return ", ".join(str(parts[f]) for f in PAST_FIELDS)
