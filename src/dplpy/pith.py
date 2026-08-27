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

# Title: pith.py
# Project: OpenDendro dplPy
# Description: Pith-offset helpers, ports of dplR's po.to.wc() and wc.to.po().
#              A pith offset is the estimated number of rings from the innermost
#              measured ring to the pith (the cambial age of the first measured
#              ring); it is what age-based Regional Curve Standardisation needs
#              to align every series on a common cambial-age axis. These two
#              functions convert between the pith-offset representation and the
#              (TRIDAS-style) wood-completeness representation.

import numpy as np
import pandas as pd
from ._validate import _require_dataframe


def po_to_wc(po: pd.DataFrame) -> pd.DataFrame:
    """Convert pith offset to (partial) wood completeness (dplR's po.to.wc()).

    The number of missing heartwood rings is the pith offset minus one.

    Parameters
    ----------
    po : pandas.DataFrame
        pith-offset table with a ``series`` column (series IDs) and a
        ``pith_offset`` column (integer years from the start of the core to the
        pith, minimum 1).

    Returns
    -------
    pandas.DataFrame
        one column, ``n_missing_heartwood`` (integer), indexed by series.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/po.to.wc.html
    """
    _require_dataframe(po)
    series = po["series"] if "series" in po.columns else po.iloc[:, 0]
    pith_offset = po["pith_offset"] if "pith_offset" in po.columns else po.iloc[:, 1]
    return pd.DataFrame(
        {"n_missing_heartwood": pd.array(np.asarray(pith_offset) - 1, dtype="Int64")},
        index=pd.Index(np.asarray(series), name="series"),
    )


def wc_to_po(wc: pd.DataFrame) -> pd.DataFrame:
    """Convert wood completeness to pith offset (dplR's wc.to.po()).

    The pith offset is the number of missing heartwood rings plus the number of
    unmeasured inner rings plus one, for every series whose pith presence is
    known (``complete``/``incomplete``) or whose missing-heartwood count is
    given; other series get a missing pith offset.

    Parameters
    ----------
    wc : pandas.DataFrame
        wood-completeness table indexed by series, with any of the optional
        columns ``pith_presence`` (``"complete"``/``"incomplete"``),
        ``n_missing_heartwood`` (int) and ``n_unmeasured_inner`` (int).

    Returns
    -------
    pandas.DataFrame
        a pith-offset table with ``series`` and ``pith_offset`` columns
        (nullable integer), suitable as input to RCS.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/wc.to.po.html
    """
    _require_dataframe(wc)
    n = len(wc)
    nan = pd.Series(np.full(n, np.nan))
    pith = wc["pith_presence"].reset_index(drop=True) if "pith_presence" in wc.columns else nan
    missing = wc["n_missing_heartwood"].reset_index(drop=True) if "n_missing_heartwood" in wc.columns else nan
    unmeasured = wc["n_unmeasured_inner"].reset_index(drop=True) if "n_unmeasured_inner" in wc.columns else nan

    known = pith.isin(["complete", "incomplete"]) | missing.notna()
    # rowSums(cbind(missing, unmeasured), na.rm=TRUE) + 1, only for known rows
    total = missing.fillna(0).to_numpy() + unmeasured.fillna(0).to_numpy() + 1
    pith_offset = pd.array([pd.NA] * n, dtype="Int64")
    pith_offset[known.to_numpy()] = np.asarray(total[known.to_numpy()], dtype="int64")

    return pd.DataFrame({"series": np.asarray(wc.index), "pith_offset": pith_offset})
