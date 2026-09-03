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

# Title: site_metadata.py
# Description: One site-metadata container that dplPy's exporters share, so the
#   site facts a file needs (id, name, species, coordinates, elevation,
#   investigators) live in a single place regardless of the output format. It is
#   populated automatically from the metadata dplPy captures when it reads a
#   Tucson file (``df.attrs['dplpy_metadata']`` / ``dpl.metadata()``), and can be
#   built or amended by hand. Today it feeds the Tucson chronology (.crn) writer;
#   the LiPD exporter (planned) will consume the same object.
#
# example usage:
#   >>> import dplpy as dpl
#   >>> rwl  = dpl.readers("ca533.rwl")
#   >>> meta = dpl.SiteMetadata.from_rwl(rwl)      # auto-filled from the header
#   >>> meta.investigators = "Lamarche, V.C."       # amend anything by hand
#   >>> dpl.writers(chron, "ca533", "crn", header=meta)

from dataclasses import dataclass, fields as _dc_fields
from typing import Optional

import pandas as pd


# A ``SiteMetadata`` is a small, typed record (a dataclass -- Python's answer to
# a MATLAB struct with named fields, but with types and defaults). Every field
# defaults to None so a partially-known site is still representable. The field
# NAMES match the keys dplPy already uses in ``df.attrs['dplpy_metadata']`` so
# auto-population is a straight copy.
@dataclass
class SiteMetadata:
    """Site/sample metadata shared by dplPy's file exporters.

    Field names mirror ``dpl.metadata()`` / ``df.attrs['dplpy_metadata']``. Use
    :meth:`from_rwl` to auto-fill from a frame read by ``dpl.readers()``,
    :meth:`from_metadata` from a metadata dict, or construct one directly and set
    fields by hand. :meth:`to_crn_header` renders the dict the Tucson ``.crn``
    writer expects (which uses a few different key names).
    """

    site_id: Optional[str] = None
    site_name: Optional[str] = None
    species_code: Optional[str] = None
    species_name: Optional[str] = None
    country_region: Optional[str] = None
    elevation_m: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    first_year: Optional[int] = None
    last_year: Optional[int] = None
    investigators: Optional[str] = None

    # -- constructors -------------------------------------------------------- #
    @classmethod
    def from_metadata(cls, meta: dict) -> "SiteMetadata":
        """Build from a metadata dict (``dpl.metadata()`` output or the same
        shape). Unknown keys (e.g. ``n_header_lines``) are ignored."""
        if not isinstance(meta, dict):
            raise TypeError("from_metadata expects a dict, not "
                            + str(type(meta)))
        known = {f.name for f in _dc_fields(cls)}
        return cls(**{k: v for k, v in meta.items() if k in known})

    @classmethod
    def from_rwl(cls, data) -> "SiteMetadata":
        """Build from a frame read by ``dpl.readers()`` (uses
        ``df.attrs['dplpy_metadata']``), or from a metadata dict directly."""
        if isinstance(data, pd.DataFrame):
            meta = data.attrs.get("dplpy_metadata")
            if meta is None:
                raise ValueError(
                    "this DataFrame has no 'dplpy_metadata' -- read the file "
                    "with dpl.readers() (which captures the header), or pass a "
                    "metadata dict / build a SiteMetadata by hand.")
            return cls.from_metadata(meta)
        if isinstance(data, dict):
            return cls.from_metadata(data)
        raise TypeError("from_rwl expects a DataFrame from dpl.readers() or a "
                        "metadata dict, not " + str(type(data)))

    @classmethod
    def coerce(cls, obj) -> "SiteMetadata":
        """Return ``obj`` if it is already a SiteMetadata, or build one from a
        metadata dict. Used by writers so they accept either form."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_metadata(obj)
        raise TypeError("expected a SiteMetadata or a metadata dict, not "
                        + str(type(obj)))

    # -- renderers ----------------------------------------------------------- #
    def to_crn_header(self) -> dict:
        """Render the header dict the Tucson ``.crn`` writer expects. The writer
        uses a few different key names (``state_country``/``species``/
        ``elevation``); this maps them. Fields that are None are omitted, so the
        writer's own "missing required key" check reports what still needs
        filling rather than writing the literal 'None'."""
        mapped = {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "species_code": self.species_code,
            "state_country": self.country_region,
            "species": self.species_name,
            "elevation": self.elevation_m,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "investigators": self.investigators,
        }
        return {k: v for k, v in mapped.items() if v is not None}
