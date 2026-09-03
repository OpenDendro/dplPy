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

# Title: combine.py
# Description: Merge several ring-width datasets into one, aligned on the union of
#   their years. A thin wrapper over pandas (dplPy datasets are already
#   year-indexed DataFrames, so this is essentially a column-wise concat), matching
#   dplR's combine.rwl(): the combined frame spans a contiguous min..max year range
#   (gap years padded with NaN), and every series column is kept side by side --
#   duplicate series IDs are not merged or renamed, exactly as in dplR.
#
# example usage:
#   >>> import dplpy as dpl
#   >>> a = dpl.readers("site_a.rwl")
#   >>> b = dpl.readers("site_b.rwl")
#   >>> both = dpl.combine_rwl([a, b])      # or dpl.combine_rwl(a, b)

import warnings

import pandas as pd


def combine_rwl(x, y=None, warn_duplicates=True) -> pd.DataFrame:
    """Combine ring-width datasets on the union of their years (dplR ``combine.rwl``).

    Parameters
    ----------
    x : pandas.DataFrame or list/tuple of DataFrames
        Either the first dataset (with ``y`` the second), or a list/tuple of
        datasets to combine in order.
    y : pandas.DataFrame, optional
        The second dataset, when ``x`` is a single DataFrame.
    warn_duplicates : bool, default True
        Emit a warning if the combined frame ends up with duplicate series names
        (the columns are still kept, matching dplR -- this only flags them).

    Returns
    -------
    pandas.DataFrame
        All series side by side, indexed by a contiguous year range spanning the
        earliest to the latest year of the inputs (years with no data are NaN).

    Notes
    -----
    Series columns are concatenated as-is: like dplR, duplicate series IDs are
    neither merged nor renamed. Deduplicate or rename beforehand if that matters
    for your workflow.
    """
    if isinstance(x, (list, tuple)):
        frames = list(x)
    elif isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        frames = [x, y]
    else:
        raise TypeError("combine_rwl expects a list of DataFrames, or two "
                        "DataFrames (x, y).")

    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError("nothing to combine: no DataFrames supplied.")
    for f in frames:
        if not isinstance(f, pd.DataFrame):
            raise TypeError("every item to combine must be a pandas DataFrame, "
                            "not " + str(type(f)) + ".")
    if len(frames) == 1:
        combined = frames[0].copy()
    else:
        combined = pd.concat(frames, axis=1)

    # contiguous year span (dplR fills interior gap years with NaN)
    index_name = next((f.index.name for f in frames if f.index.name), "Year")
    lo, hi = int(combined.index.min()), int(combined.index.max())
    combined = combined.reindex(range(lo, hi + 1))
    combined.index.name = index_name

    if warn_duplicates:
        dups = combined.columns[combined.columns.duplicated()].unique().tolist()
        if dups:
            warnings.warn("combined dataset has duplicate series names (kept "
                          "as-is): " + ", ".join(str(d) for d in dups))
    return combined
