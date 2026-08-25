import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _read_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


# dplR common.interval() reference dimensions (validated against dplR 1.7.9):
#   (n_series, n_years, first_year, last_year)
_DPLR_REF = {
    ("co021", "series"): (33, 288, 1660, 1947),
    ("co021", "years"):  (27, 458, 1490, 1947),
    ("co021", "both"):   (28, 435, 1528, 1962),
    ("ca533", "series"): (30, 242, 1727, 1968),
    ("ca533", "years"):  (16, 722, 1247, 1968),
    ("ca533", "both"):   (22, 500, 1471, 1970),
}


@pytest.mark.parametrize("dataset,ctype", list(_DPLR_REF.keys()))
def test_common_interval_matches_dplR(dataset, ctype):
    data = _read_quiet("tests/data/csv/%s.csv" % dataset)
    ci = dpl.common_interval(data, type=ctype)
    n_series, n_years, y0, y1 = _DPLR_REF[(dataset, ctype)]
    assert ci.shape == (n_years, n_series)
    assert int(ci.index.min()) == y0 and int(ci.index.max()) == y1
    # the result is a complete rectangle: no missing values anywhere
    assert not ci.isna().any().any()


def test_common_interval_bad_type():
    data = _read_quiet("tests/data/csv/co021.csv")
    with pytest.raises(ValueError):
        dpl.common_interval(data, type="bogus")


def test_common_interval_bad_input():
    with pytest.raises(TypeError):
        dpl.common_interval("not a frame")


def test_common_interval_rescues_empty_common_period():
    # On unevenly-distributed co021 the strict all-series intersection is empty,
    # so rwi_stats(period='common') is NaN. A common_interval selection restores
    # a valid, complete block and hence a real rbar/EPS.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(_read_quiet("tests/data/csv/co021.csv"),
                              fit="spline", plot=False)
            empty = dpl.rwi_stats(rwi, period="common")
            both = dpl.rwi_stats(rwi, common_interval="both")
            span = dpl.rwi_stats(rwi, common_interval=(1600, 1900))
    assert np.isnan(empty["rbar_eff"][0])                 # the original problem
    assert both["n_cores"][0] == 28                       # dplR 'both' -> 28 series
    assert np.isfinite(both["rbar_eff"][0])
    # user-specified span keeps only series with data across it
    assert span["n_cores"][0] >= 2 and np.isfinite(span["rbar_eff"][0])


def test_sss_common_interval_matches_dplR_composition():
    # dplR's recipe for a common-interval SSS is sss(common.interval(rwi)); the
    # sss(common_interval=...) convenience must reproduce that exactly, and SSS
    # must stay <= 1 across the trimmed block.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(_read_quiet("tests/data/csv/co021.csv"),
                              fit="spline", plot=False)
            via_param = dpl.sss(rwi, common_interval="both")
            via_compose = dpl.sss(dpl.common_interval(rwi, type="both"))
            full = dpl.sss(rwi)
    assert list(via_param.index) == list(via_compose.index)
    assert np.allclose(via_param.to_numpy(), via_compose.to_numpy())
    # trimmed to the 'both' interval (1528-1962), and SSS <= 1 there
    assert int(via_param.index.min()) == 1528 and int(via_param.index.max()) == 1962
    assert (via_param <= 1 + 1e-9).all()
    # the no-interval default still spans the full record (dplR behaviour)
    assert int(full.index.min()) == 1176 and int(full.index.max()) == 1963


def test_rwi_stats_bad_common_interval():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rwi = dpl.detrend(_read_quiet("tests/data/csv/co021.csv"),
                              fit="spline", plot=False)
    with pytest.raises(ValueError):
        dpl.rwi_stats(rwi, common_interval="nonsense")
