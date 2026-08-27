import warnings

import numpy as np
import pytest
import dplpy as dpl

# Full-pipeline detrend() over every fit type on ca533. RWI series should keep
# the data's shape, be all-positive (ratio) and centre near 1; a couple of values
# are pinned against the current dplR-validated output to catch silent drift.


def _ca533():
    return dpl.readers("./tests/data/csv/ca533.csv")


def test_detrend_all_fits_ratio():
    data = _ca533()
    fits = {}
    for fit in ("spline", "ModNegEx", "Hugershoff", "linear", "horizontal"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[fit] = dpl.detrend(data, fit=fit, method="ratio", plot=False)
    for fit, rwi in fits.items():
        assert rwi.shape == data.shape, fit
        assert list(rwi.columns) == list(data.columns), fit
        vals = rwi.to_numpy()
        vals = vals[~np.isnan(vals)]
        assert np.all(np.isfinite(vals)), fit
        # spline / ModNegEx / Hugershoff / horizontal fit strictly positive
        # curves, so their ratio RWI stays positive and centres on ~1. A linear
        # fit can cross zero, so its ratio legitimately goes negative -- exclude
        # it from the positivity/centring checks.
        if fit != "linear":
            assert np.all(vals > 0), fit
            assert 0.9 < vals.mean() < 1.1, fit
    # pinned RWI values (CAM011's first ring, 626)
    assert fits["spline"]["CAM011"].dropna().iloc[0] == pytest.approx(1.196322, abs=1e-4)
    assert fits["linear"]["CAM011"].dropna().iloc[0] == pytest.approx(1.592794, abs=1e-4)


def test_detrend_all_fits_difference():
    data = _ca533()
    for fit in ("spline", "ModNegEx", "Hugershoff", "linear", "horizontal"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rwi = dpl.detrend(data, fit=fit, method="difference", plot=False)
        assert rwi.shape == data.shape, fit
        vals = rwi.to_numpy()
        vals = vals[~np.isnan(vals)]
        # difference RWI is centred on ~0
        assert abs(vals.mean()) < 0.1, fit
