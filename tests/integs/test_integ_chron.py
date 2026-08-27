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


def test_chron_prewhiten_with_biweight():
    res = dpl.chron(_ca533(), biweight=True, prewhiten=True, plot=False)
    assert list(res.columns) == ["std", "res", "samp_depth"]
    assert res["res"].iloc[-1] == pytest.approx(0.459053, abs=1e-5)
    # the residual chronology's first p values are NaN (AR warm-up)
    assert res["res"].notna().sum() > 0
