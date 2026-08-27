import numpy as np

from dplpy.tbrm import tbrm, tbrm_rows


def test_tbrm():
    assert tbrm([-72, 2, 2, 2]) == 2


def test_tbrm_rows_matches_scalar_tbrm_per_row():
    # tbrm_rows must reproduce tbrm(row_without_nan) row-for-row to machine
    # precision -- the property that lets the hot paths use it in place of a
    # per-row Python loop over tbrm.
    rng = np.random.RandomState(0)
    mat = rng.randn(40, 12) * 1.5 + 3.0
    # sprinkle NaNs (including a fully-NaN row) to exercise the NaN handling
    mat[rng.rand(*mat.shape) < 0.25] = np.nan
    mat[7, :] = np.nan
    got = tbrm_rows(mat)
    for t in range(mat.shape[0]):
        row = mat[t][~np.isnan(mat[t])]
        if row.size == 0:
            assert np.isnan(got[t])
        else:
            assert abs(got[t] - tbrm(row, c=9)) < 1e-12


def test_tbrm_rows_all_nan_row_is_nan():
    mat = np.array([[np.nan, np.nan], [1.0, 3.0]])
    out = tbrm_rows(mat)
    assert np.isnan(out[0]) and np.isfinite(out[1])
