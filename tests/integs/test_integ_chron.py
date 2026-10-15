import numpy as np
import pytest
import dplpy as dpl

# Full-pipeline chron() on ca533. Values are pinned (regression) against the
# current dplR-validated output so a silent numeric change is caught, not just a
# crash. ca533 spans 626-1983 with a peak sample depth of 30.


def _ca533():
    return dpl.readers("./tests/data/csv/ca533.csv")


def test_chron_no_prewhiten_no_biweight():
    res = dpl.chron(_ca533(), biweight=False, prewhiten=False, plot=False)
    assert list(res.columns) == ["std", "samp_depth"]
    assert res.shape == (1358, 2)
    assert res.index.name == "Year"
    assert int(res.index.min()) == 626 and int(res.index.max()) == 1983
    assert int(res["samp_depth"].max()) == 30
    # arithmetic mean of the (single) earliest series and the full-depth end year
    assert res["std"].iloc[0] == pytest.approx(0.17, abs=1e-6)
    assert res["std"].iloc[-1] == pytest.approx(0.66, abs=1e-6)


def test_chron_prewhiten_no_biweight():
    res = dpl.chron(_ca533(), biweight=True, prewhiten=False, plot=False)
    assert list(res.columns) == ["std", "samp_depth"]
    # biweight mean differs from the arithmetic mean once depth > 1
    assert res["std"].iloc[-1] == pytest.approx(0.638166, abs=1e-5)


def test_chron_prewhiten_with_biweight_dplr_rule():
    # dplR-validated: with the global-AIC-minimum rule chron() prewhitens with
    # Yule-Walker (order.max=10), matching dplR chron(prewhiten=TRUE,
    # method="yule-walker", order.max=10) to <1e-6.
    res = dpl.chron(_ca533(), biweight=True, prewhiten=True, plot=False,
                    first_aic_min=False)
    assert list(res.columns) == ["std", "res", "samp_depth"]
    assert res["res"].iloc[-1] == pytest.approx(0.466448, abs=1e-5)
    # the residual chronology's first p values are NaN (AR warm-up)
    assert res["res"].notna().sum() > 0
    assert np.isnan(res["res"].iloc[0])


def test_chron_prewhiten_with_biweight_default_first_aic_min():
    # package default: first local AIC minimum (ARSTAN's rule); the residual
    # chronology differs from the dplR global-minimum rule (regression pin)
    res = dpl.chron(_ca533(), biweight=True, prewhiten=True, plot=False)
    assert res["res"].iloc[-1] == pytest.approx(0.460928, abs=1e-5)
    assert res.shape == (1358, 3)                       # full year index kept
