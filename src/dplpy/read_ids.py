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

# Title: read_ids.py
# Project: OpenDendro dplPy
# Description: Derives tree and core identifiers from tree-ring series names
#              following the site-tree-core (STC) naming convention, producing
#              the id mapping consumed by dpl.rwi_stats()/dpl.sss() to group
#              cores by tree. This is the deterministic subset of dplR's
#              read.ids(): a pattern-based default plus an explicit fixed-width
#              STC character mask. It deliberately does NOT attempt dplR's
#              fuzzy autoread.ids() heuristics (typo correction, look-alike
#              character substitution, etc.) for unconventional naming.
#
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> ids = dpl.read_ids(data)                 # letter-core names, e.g. ABC001A
# >>> ids = dpl.read_ids(data, stc=(3, 2, 1))  # digit-core names, e.g. CAM031

import re
import warnings

import pandas as pd

# leading letters (site) + digit run (tree number) + trailing remainder (core)
_STC_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)([A-Za-z0-9]*)$")


def read_ids(data, stc=None):
    """Derive tree and core IDs from series names (site-tree-core convention).

    Extended Summary
    ----------------
    Tree-ring series are conventionally named with a site code, a tree number,
    and a core designator (the "STC" convention). This function splits each
    series name into a tree identifier (site + tree number, so that different
    trees -- and different sites -- never collide) and a core designator, and
    returns the mapping used by dpl.rwi_stats() and dpl.sss() to know which
    cores belong to the same tree.

    Two parsing modes are available:

    Pattern default (stc=None). Each name is split on its letter/digit
    boundaries: leading letters are the site, the first run of digits is the
    tree number, and any trailing characters are the core. This handles names
    with a LETTER core and a variable-length site automatically, e.g. both
    "ABC001A" (site ABC, tree 001, core A -> tree "ABC001") and "ABCD01"
    (site ABCD, tree 01, no core -> tree "ABCD01").

    IMPORTANT: this pattern cannot see a DIGIT core. A name like "CAM031"
    (site CAM, tree 03, core 1) still matches -- the whole digit run "031" is
    read as the tree number and the core comes out empty -- so it parses
    WITHOUT any warning but leaves "CAM031" and "CAM032" as two separate trees
    instead of grouping them under "CAM03". For digit cores (and for site codes
    that contain digits) you MUST supply an explicit stc mask. The only names
    that trigger the "could not parse" warning are those that do not match the
    pattern at all (no leading letters, embedded separators, etc.).

    STC mask (stc=(site_len, tree_len[, core_len])). Splits each name by fixed
    character positions: the tree identifier is the first site_len+tree_len
    characters, and the core is the remainder (or exactly core_len characters
    if a third element is given). Use this whenever the pattern default cannot
    resolve the structure -- e.g. digit cores like "CAM031" with stc=(3, 2, 1),
    or site codes that contain digits.

    Parameters
    ----------
    data : pandas dataframe or iterable of str
        a dataframe whose column names are the series names (such as from
        dpl.readers()), or any iterable of series-name strings.
    stc : tuple of int, or None, default None
        None to use the pattern default; otherwise a (site_len, tree_len) or
        (site_len, tree_len, core_len) character mask.

    Returns
    -------
    ids : pandas dataframe indexed by series name, with columns 'tree' and
        'core'. Pass this directly as the `ids` argument of dpl.rwi_stats(),
        dpl.rwi_stats_running(), or dpl.sss().

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> ids = dpl.read_ids(data)                 # letter-core names
    >>> ids = dpl.read_ids(data, stc=(3, 2, 1))  # digit-core names (e.g. CAM031)
    >>> dpl.rwi_stats(dpl.detrend(data, plot=False), ids=ids)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/read.ids.html

    """
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
    else:
        names = [str(c) for c in data]

    if len(names) == 0:
        raise ValueError("No series names found to parse.")

    if stc is None:
        trees, cores, unparsed = _parse_pattern(names)
        if unparsed:
            warnings.warn(
                "read_ids could not parse these series names with the default "
                "pattern (each was treated as its own single-core tree): "
                + ", ".join(unparsed)
                + ". If they use a digit core or a site code containing digits, "
                "supply an stc=(site_len, tree_len, core_len) mask instead.\n"
            )
    else:
        _validate_stc(stc)
        trees, cores = _parse_stc(names, stc)

    return pd.DataFrame(
        {"tree": trees, "core": cores},
        index=pd.Index(names, name="series"),
    )


def _parse_pattern(names):
    trees, cores, unparsed = [], [], []
    for nm in names:
        m = _STC_PATTERN.match(nm)
        if m:
            site, tree_num, core = m.groups()
            trees.append(site + tree_num)
            cores.append(core)
        else:
            # can't resolve -> treat the whole name as its own single-core tree
            trees.append(nm)
            cores.append("")
            unparsed.append(nm)
    return trees, cores, unparsed


def _validate_stc(stc):
    if not (isinstance(stc, (tuple, list)) and len(stc) in (2, 3)):
        raise ValueError(
            "stc must be a tuple of (site_len, tree_len) or "
            "(site_len, tree_len, core_len), got " + repr(stc) + "."
        )
    if not all(isinstance(x, int) and x >= 0 for x in stc):
        raise ValueError("stc entries must be non-negative integers.")
    if stc[0] + stc[1] < 1:
        raise ValueError("site_len + tree_len must be at least 1.")


def _parse_stc(names, stc):
    prefix = stc[0] + stc[1]           # tree id = site + tree number
    core_len = stc[2] if len(stc) == 3 else None
    trees, cores = [], []
    for nm in names:
        trees.append(nm[:prefix])
        if core_len is None:
            cores.append(nm[prefix:])
        else:
            cores.append(nm[prefix:prefix + core_len])
    return trees, cores
