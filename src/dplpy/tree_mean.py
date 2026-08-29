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

# Title: tree_mean.py
# Project: OpenDendro dplPy
# Description: Average the cores of each tree into a single tree-level series --
#              a port of dplR 1.7.9 R/treeMean.R (dplR's `treeMean`). Given a
#              ring-width dataset (years x cores) and the tree/core structure
#              from dpl.read_ids(), this returns a new dataset (years x TREES)
#              where every tree's value in a year is the mean of its cores.
#
# Why: building a chronology from tree-level means (rather than straight from
# cores) can raise the common signal for trees with odd growth habits or poor
# circuit uniformity -- each tree contributes once, so a tree cored many times
# does not dominate. Typical use is dpl.chron(dpl.tree_mean(rwi, ids)).
#
# Fidelity note: dplR's default is na.rm=FALSE, so a tree's value for a year is
# NA unless *every* one of its cores has a measurement that year. dplPy keeps
# that default; pass na_rm=True to average whatever cores are present (usually
# what you want when the tree means feed a chronology).
#
# example usage:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/ca533.csv")
# >>> ids  = dpl.read_ids(data)
# >>> trees = dpl.tree_mean(data, ids)          # years x trees
# >>> crn  = dpl.chron(dpl.tree_mean(dpl.detrend(data), ids, na_rm=True))

import warnings

import numpy as np
import pandas as pd

from ._validate import _require_dataframe
from .rwi_stats import _resolve_tree_mapping     # shared {series: tree} resolver


def tree_mean(rwl, ids, na_rm=False):
    """Average each tree's cores into a single tree-level series (dplR ``treeMean``).

    Parameters
    ----------
    rwl : pandas.DataFrame
        A ring-width dataset (years x cores/series).
    ids : pandas.DataFrame or dict
        The tree/core structure that says which tree each series belongs to --
        the output of :func:`read_ids` (a DataFrame indexed by series name with
        a ``tree`` column), or a plain ``{series_name: tree_id}`` dict. Series
        are matched to trees by name, so ``ids`` need not be in column order.
    na_rm : bool, default False
        If False (dplR's default), a tree's value for a year is NaN unless every
        one of its cores has a measurement that year. If True, the mean is taken
        over whatever cores are present (a year is NaN only when the tree has no
        core at all that year).

    Returns
    -------
    pandas.DataFrame
        A ring-width dataset (years x trees). Columns are the unique tree IDs, in
        order of first appearance among the input columns; the year index is
        preserved. Feed it straight to :func:`detrend` / :func:`chron`.

    Examples
    --------
    >>> ids = dpl.read_ids(data)
    >>> dpl.tree_mean(data, ids)
    >>> dpl.chron(dpl.tree_mean(dpl.detrend(data), ids, na_rm=True))

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#tree_mean
    """
    frame = _require_dataframe(rwl)
    if not isinstance(na_rm, bool):
        raise ValueError("'na_rm' must be either True or False")

    cols = list(frame.columns)
    tree_of = _resolve_tree_mapping(ids, cols)          # {series: tree}, covers all cols

    trees_in_order = [tree_of[c] for c in cols]
    if any(pd.isna(t) for t in trees_in_order):
        raise ValueError("missing tree IDs are not allowed")
    u_trees = list(dict.fromkeys(trees_in_order))       # unique, first-appearance order

    X = frame.to_numpy(dtype=float)                     # (n_years, n_cores)
    res = np.full((X.shape[0], len(u_trees)), np.nan)
    for i, t in enumerate(u_trees):
        take = [j for j, c in enumerate(cols) if tree_of[c] == t]
        sub = X[:, take]
        if na_rm:
            # a year with no present core is an all-NaN slice -> NaN (not 0);
            # numpy warns on that, so silence just this call.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res[:, i] = np.nanmean(sub, axis=1)
        else:
            res[:, i] = np.mean(sub, axis=1)            # any NaN core -> NaN, as in dplR

    return pd.DataFrame(res, index=frame.index, columns=u_trees)
