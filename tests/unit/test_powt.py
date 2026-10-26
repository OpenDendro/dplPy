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


# dplR powt(method='cook') per-series powers for the first few ca533 series
# (dplR 1.7.9). Cook is deterministic OLS, so dplPy reproduces these exactly.
_DPLR_COOK_CA533 = {
    "CAM011": 0.71236143893634,
    "CAM021": 0.770498860140902,
    "CAM031": 0.675286368446009,
    "CAM032": 0.837457599335083,
    "CAM041": 0.589542905376192,
}


def test_powt_cook_matches_dplR_powers():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    pt, power = dpl.powt(rwl, method="cook", return_power=True)
    for series, expected in _DPLR_COOK_CA533.items():
        assert power[series] == pytest.approx(expected, abs=1e-6)
    # transform is x**p at the series' own power
    x = rwl["CAM011"].dropna().to_numpy()
    assert np.allclose(pt["CAM011"].dropna().to_numpy(),
                       x ** power["CAM011"], atol=1e-10)


def test_powt_cook_default_method():
    # cook is the default (unlike dplR, whose default is 'universal')
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    _, p_default = dpl.powt(rwl, return_power=True)
    _, p_cook = dpl.powt(rwl, method="cook", return_power=True)
    assert (p_default == p_cook).all()


def test_powt_cook_vector():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    s = rwl["CAM011"]
    out, p = dpl.powt(s, method="cook", return_power=True)
    assert isinstance(out, pd.Series)
    assert p == pytest.approx(0.7123613, abs=1e-6)


def test_powt_rescale_restores_moments():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    rescaled = dpl.powt(rwl, method="cook", rescale=True)
    col = "CAM011"
    orig = rwl[col].to_numpy()
    got = rescaled[col].to_numpy()
    assert np.nanmean(got) == pytest.approx(np.nanmean(orig), abs=1e-8)
    assert np.nanstd(got, ddof=1) == pytest.approx(np.nanstd(orig, ddof=1), abs=1e-8)


def test_powt_universal_matches_dplR():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = dpl.powt(rwl, method="universal", return_power=True)
    # dplR lme4 estimate is 0.568198; the statsmodels optimizer matches to ~1e-5
    assert p == pytest.approx(0.568198, abs=1e-4)


def test_powt_universal_vector_rejected():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        dpl.powt(rwl["CAM011"], method="universal")


def test_powt_negative_values_rejected():
    df = pd.DataFrame({"A": [1.0, 2.0, -1.0]}, index=pd.Index([1, 2, 3], name="Year"))
    with pytest.raises(ValueError):
        dpl.powt(df)


def test_powt_bad_method():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        dpl.powt(rwl, method="bogus")


def test_powt_bad_input():
    with pytest.raises(TypeError):
        dpl.powt("not a frame")


# --- diagnostic stats + plot (ARSTAN-style; a dplPy addition) ---------------
def test_powt_return_stats_shape_and_columns():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    pt, stats = dpl.powt(rwl, return_stats=True)
    assert isinstance(pt, pd.DataFrame)
    assert list(stats.columns) == ["skew_before", "skew_after", "r_before",
                                   "r_after", "power"]
    assert len(stats) == rwl.shape[1]                 # one row per series


def test_powt_transform_reduces_skew_and_spread_level_corr():
    # the whole point of the diagnostic: after the transform, both the skew and
    # the spread-vs-level correlation shrink toward zero (median over series)
    rwl = _read_quiet("tests/data/csv/co021.csv")
    _, stats = dpl.powt(rwl, return_stats=True)
    assert stats["skew_after"].abs().median() < stats["skew_before"].abs().median()
    assert stats["r_after"].abs().median() < stats["r_before"].abs().median()


def test_powt_return_power_and_stats_tuple():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    out = dpl.powt(rwl, return_power=True, return_stats=True)
    assert isinstance(out, tuple) and len(out) == 3   # (data, power, stats)
    data, power, stats = out
    # the stats 'power' column matches the returned power Series
    assert np.allclose(stats["power"].to_numpy(),
                       power.reindex(stats.index).to_numpy(), equal_nan=True)


def test_powt_universal_stats_constant_power():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    _, stats = dpl.powt(rwl, method="universal", return_stats=True)
    assert stats["power"].nunique() == 1              # one power for all series


def test_powt_plot_smoke():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from unittest.mock import patch
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    plt.close("all")
    with patch("matplotlib.pyplot.show") as show:
        dpl.powt(rwl, plot=True)
    show.assert_called_once()
    # F2: under a non-interactive backend the figure is closed after display
    assert len(plt.get_fignums()) == 0
    plt.close("all")


def test_powt_drops_all_nan_columns_and_is_case_insensitive():
    rwl = _read_quiet("tests/data/csv/ca533.csv").loc[1900:1983]
    empty = [c for c in rwl.columns if rwl[c].isna().all()]
    assert empty
    with pytest.warns(UserWarning, match="no measured values"):
        out = dpl.powt(rwl)
    assert set(out.columns) == set(rwl.columns) - set(empty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = dpl.powt(rwl, method="Cook")
        b = dpl.powt(rwl, method="cook")
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_powt_output_attrs():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    rwl.attrs.update({"dplpy_precision": 0.01, "kind": "rwl", "site": "ca533"})
    out = dpl.powt(rwl)
    assert out.attrs["kind"] == "rwl_powt" and out.attrs["site"] == "ca533"
    assert "dplpy_precision" not in out.attrs
    pt, p = dpl.powt(rwl, return_power=True)
    assert pt.attrs["kind"] == "rwl_powt"


# ----- method='druckenbrod' and the inverse (added with disturbance_detrend) -----

def test_powt_druckenbrod_forward_and_inverse_roundtrip():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    s = rwl["CAM011"].dropna()
    pt, p = dpl.powt(s, method="druckenbrod", return_power=True)
    # forward is log(x) (p<=0.05) or x**p, matching v267_rga's branch
    x = s.to_numpy()
    expect = np.log(x) if p <= 0.05 else x ** (1.0 if p > 1 else p)
    assert np.allclose(pt.to_numpy(), expect, atol=1e-12)
    # inverse re-expresses to the original
    back = dpl.powt(pt, method="druckenbrod", inverse=True, power=p)
    assert np.allclose(back.to_numpy(), x, atol=1e-9)


def test_powt_druckenbrod_differs_from_cook():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    _, p_cook = dpl.powt(rwl, method="cook", return_power=True)
    _, p_dan = dpl.powt(rwl, method="druckenbrod", return_power=True)
    # cook is |1-b| (never negative); druckenbrod is signed 1-b and can be
    assert (p_cook >= 0).all()
    assert not np.allclose(p_cook.to_numpy(), p_dan.reindex(p_cook.index).to_numpy())


def test_powt_cook_inverse_roundtrip_dataframe():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    pt, power = dpl.powt(rwl, method="cook", return_power=True)
    back = dpl.powt(pt, method="cook", inverse=True, power=power)
    # round-trips where defined (cook power > 0, so x**p -> x**(1/p))
    a = back.to_numpy(); b = rwl[back.columns].to_numpy()
    m = ~np.isnan(b)
    assert np.allclose(a[m], b[m], atol=1e-8)


def test_powt_inverse_requires_power():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        dpl.powt(rwl, method="cook", inverse=True)


def test_powt_druckenbrod_allowed_on_series():
    rwl = _read_quiet("tests/data/csv/ca533.csv")
    out = dpl.powt(rwl["CAM011"].dropna(), method="druckenbrod")
    assert isinstance(out, pd.Series)
