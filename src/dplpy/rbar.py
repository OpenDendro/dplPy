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

# Date: 5/12/2023
# Author: Ifeoluwa Ale
# Title: rbar.py
# Description: Mean interseries correlation (rbar) helpers: the one
#              pairwise-correlation-with-minimum-overlap matrix the signal
#              statistics share (audit E19), and the moving-window running
#              rbar used for variance stabilization (audit E14 / F5).

import numpy as np

# NOTE: the earlier rectangle-maximising common_interval() that lived here has
# been superseded by dplpy.common_interval (see common_interval.py), a faithful
# port of dplR's common.interval() offering the 'series', 'years' and 'both'
# selection strategies.


def pairwise_corr_matrix(data, method="pearson", min_overlap=None, strict=False):
    """Pairwise-complete correlation matrix between the columns of ``data`` with
    a minimum-overlap requirement, as a numpy array with a NaN diagonal.

    When ``min_overlap`` is given, a series pair is kept only if its number of
    overlapping (both-present) years passes the threshold -- strictly greater
    (``strict=True``, ARSTAN's Briffa n>20) or at least (``strict=False``, the
    moving-window rbar and rwi_stats' ``min_corr_overlap``); other pairs are
    NaN. The single home for this computation (audit E19): rwi_stats, sss,
    chron_stabilized and stabilize_chron all use it.
    """
    # corr.to_numpy() can be a read-only view under numpy 2 / copy-on-write, so
    # fill_diagonal needs an explicit writable copy, not the view pandas hands back.
    arr = data.corr(method).to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    if min_overlap is not None:
        presence = data.notna().to_numpy(dtype=float)
        overlap = presence.T @ presence
        keep = overlap > min_overlap if strict else overlap >= min_overlap
        arr = np.where(keep, arr, np.nan)
    return arr


def pairwise_corr_mean(data, method="pearson", min_overlap=None, strict=False):
    """Mean of the off-diagonal pairwise correlations between the columns of ``data``
    (self-correlations excluded), counting only pairs that pass ``min_overlap``
    (see `pairwise_corr_matrix()`). A flat mean over every valid off-diagonal
    entry (each retained pair weighted equally), matching dplR's
    ``mean(corMat, na.rm = TRUE)``. NaN when no pair qualifies.
    """
    arr = pairwise_corr_matrix(data, method, min_overlap=min_overlap, strict=strict)
    finite = arr[~np.isnan(arr)]
    return float(finite.mean()) if finite.size else np.nan


def _pair_rbar(n, sx, sy, sxx, syy, sxy, min_overlap):
    """Flat mean of pairwise Pearson correlations from masked pair sums (one row
    per window, one column per pair): ``n`` overlap counts, ``sx``/``sxx`` the
    sum and sum of squares of series a over the rows where b is present,
    ``sy``/``syy`` the same for b, ``sxy`` the cross-product sum."""
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    cov = n * sxy - sx * sy
    # a constant series over the overlap has zero variance (pandas gives NaN);
    # the closed form leaves rounding noise there, so test relative to the sum
    # of squares rather than against exactly zero.
    ok = (n >= max(min_overlap, 2)) & (vx > 1e-12 * n * sxx) & (vy > 1e-12 * n * syy)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok, cov / np.sqrt(np.where(ok, vx * vy, 1.0)), np.nan)
    cnt = ok.sum(axis=-1)
    with np.errstate(invalid="ignore"):
        out = np.where(cnt > 0, np.nansum(r, axis=-1) / np.where(cnt > 0, cnt, 1), np.nan)
    return out


# above this many (years x series-pairs) cells the all-windows cumulative-sum
# formulation would need too much memory, and the O(1)-memory sliding update is
# used instead (same results)
_CUMSUM_CELLS = 3_000_000


def running_rbar_vector(data, window_length, min_seg_ratio, n_samps=None):
    """Moving-window rbar vector (Frank et al. 2006 "RUNNINGr"), padded at the ends,
    matching dplR's chron.stabilized().

    ``data`` is a years x series DataFrame (location does not matter: the
    correlations are invariant to centring). Returns a numpy array of length
    ``nrows``: the Pearson rbar of the window of ``window_length`` years centred
    on each year (pairs overlapping fewer than ``window_length * min_seg_ratio``
    years are left out), with the leading/trailing ``window_length/2`` positions
    padded with the nearest computed value, and NaN where the total sample
    depth is 0.

    The per-pair window sums (overlap count, masked sums, sums of squares and
    cross products) come from cumulative sums along time, so every window's
    correlation matrix is formed at once in O(n_years x n_series^2) (audit F5);
    for very wide matrices a sliding rank-1 update with the same result is used
    instead. Results equal the per-window ``DataFrame.corr`` route.
    """
    X = np.asarray(data.to_numpy(dtype=float))
    num_years, s = X.shape
    w = int(window_length)
    rbar_array = np.full(num_years, np.nan)
    if n_samps is None:
        n_samps = np.count_nonzero(~np.isnan(X), axis=1)
    else:
        n_samps = np.asarray(n_samps)

    n_windows = num_years - w + 1
    if n_windows >= 1 and s >= 2:
        M = (~np.isnan(X)).astype(float)
        X0 = np.where(np.isnan(X), 0.0, X)
        X2 = X0 * X0
        min_overlap = w * min_seg_ratio
        target = w // 2 if w % 2 == 0 else (w - 1) // 2
        iu = np.triu_indices(s, k=1)

        if num_years * iu[0].size <= _CUMSUM_CELLS:
            def wsum(A, B):
                # per-window sum over rows of A[t, a] * B[t, b] for every pair (a, b)
                P = A[:, iu[0]] * B[:, iu[1]]
                C = np.empty((num_years + 1, P.shape[1]))
                C[0] = 0.0
                np.cumsum(P, axis=0, out=C[1:])
                return C[w:] - C[:n_windows]
            vals = _pair_rbar(wsum(M, M), wsum(X0, M), wsum(M, X0), wsum(X2, M),
                              wsum(M, X2), wsum(X0, X0), min_overlap)
            rbar_array[target:target + n_windows] = vals
        else:
            def anchor(i):
                m = M[i:i + w]
                x0 = X0[i:i + w]
                return (m.T @ m, x0.T @ m, X2[i:i + w].T @ m, x0.T @ x0)

            N, SX, SXX, SXY = anchor(0)
            for i in range(n_windows):
                if i > 0:
                    if i % 64 == 0:                    # re-anchor exactly
                        N, SX, SXX, SXY = anchor(i)
                    else:                              # row i-1 leaves, row i+w-1 enters
                        for row, sign in ((i - 1, -1.0), (i + w - 1, 1.0)):
                            m = M[row]
                            x0 = X0[row]
                            N += sign * np.outer(m, m)
                            SX += sign * np.outer(x0, m)
                            SXX += sign * np.outer(X2[row], m)
                            SXY += sign * np.outer(x0, x0)
                rbar_array[i + target] = _pair_rbar(
                    N[iu], SX[iu], SX.T[iu], SXX[iu], SXX.T[iu], SXY[iu], min_overlap)

    # pad the leading/trailing not-yet-computed runs with the nearest real value
    # (dplR pads with the first/last non-NA entry of movingRbarVec); a genuine
    # rbar of 0.0 is a valid value, so we detect "not computed" via NaN, not 0.
    valid = np.flatnonzero(~np.isnan(rbar_array))
    if valid.size > 0:
        rbar_array[:valid[0]] = rbar_array[valid[0]]
        rbar_array[valid[-1]:] = rbar_array[valid[-1]]

    # a year with zero total sample depth has no meaningful rbar, padded or not
    rbar_array[n_samps == 0] = np.nan
    return rbar_array
