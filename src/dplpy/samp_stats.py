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

# Title: samp_stats.py
# Project: OpenDendro dplPy
# Description: Per-year sample-tracking statistics for a ring-width dataset:
#              sample depth, mean segment length, and mean cambial age. These are
#              the "num / seg / age" columns of an ARSTAN "tabs" file, provided as
#              a helper so an author can assemble their own text output alongside
#              whatever chronologies they choose (see dpl.writers(..., "txt")).

import numpy as np
import pandas as pd
from ._validate import _require_dataframe


def samp_stats(data: pd.DataFrame):
    """Per-year sample-tracking statistics for a ring-width dataset.

    For every year in the dataset this reports how many series are present and,
    across just those present series, their mean length and mean cambial age --
    the ``num`` / ``seg`` / ``age`` columns of an ARSTAN "tabs" file.

    Parameters
    ----------
    data : pandas dataframe
        ring widths, years as the index and series as columns (e.g. from
        dpl.readers()). Raw widths or detrended indices give the same result --
        only the pattern of present/absent values is used.

    Returns
    -------
    out : pandas dataframe indexed by year with columns

        - ``samp_depth`` : number of series present that year (the sample depth;
          ARSTAN's ``num``). An integer count.
        - ``seg`` : mean segment length of the series present that year, where a
          series' segment length is its number of measured rings.
        - ``age`` : mean cambial age that year, where a series' cambial age is the
          number of years since its first ring (``year - first_year + 1``).

        Years with no series present get ``samp_depth`` 0 and NaN for ``seg`` /
        ``age``.

    Notes
    -----
    ``seg`` counts a series' measured rings; ``age`` counts years since its first
    ring. The two differ only for a series with an interior gap (an unmeasured
    year inside its span), for which the tree still ages but the ring is not
    counted.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.samp_stats(data)
    """
    _require_dataframe(data)

    present = data.notna()                              # years x series, bool
    samp_depth = present.sum(axis=1)                    # per-year count

    # each series' segment length (measured-ring count) and first year
    seg_len = present.sum(axis=0)                       # per-series ring count
    first_year = pd.Series(
        {col: data[col].first_valid_index() for col in data.columns},
        dtype="float64",
    )

    depth = samp_depth.replace(0, np.nan)               # avoid divide-by-zero
    seg = present.mul(seg_len, axis=1).sum(axis=1) / depth
    mean_first = present.mul(first_year, axis=1).sum(axis=1) / depth
    age = (data.index.to_series() + 1) - mean_first     # year - mean_first + 1

    out = pd.DataFrame(
        {"samp_depth": samp_depth.astype(int), "seg": seg, "age": age},
        index=data.index,
    )
    out.index.name = data.index.name or "year"
    return out
