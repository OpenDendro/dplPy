import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _frame():
    # interior gaps at 1991 and 1994; a real 0 at 1993 (a locally absent ring,
    # must be preserved); no leading/trailing NaN here.
    return pd.DataFrame(
        {"S": [0.2, np.nan, 0.4, 0.0, np.nan, 0.8]},
        index=pd.Index(range(1990, 1996), name="Year"),
    )


def test_fill_mean():
    out = dpl.fill_internal(_frame(), fill="Mean")
    # mean of the four present values (0.2,0.4,0.0,0.8) = 0.35
    assert list(np.round(out["S"].to_numpy(), 6)) == [0.2, 0.35, 0.4, 0.0, 0.35, 0.8]


def test_fill_linear():
    out = dpl.fill_internal(_frame(), fill="Linear")
    assert list(np.round(out["S"].to_numpy(), 6)) == [0.2, 0.3, 0.4, 0.0, 0.4, 0.8]


def test_fill_constant():
    out = dpl.fill_internal(_frame(), fill=-1)
    assert list(out["S"].to_numpy()) == [0.2, -1.0, 0.4, 0.0, -1.0, 0.8]


def test_fill_spline_fills_interior_only():
    df = pd.DataFrame(
        {"S": [np.nan, 0.5, 0.6, np.nan, 0.9, 1.0, np.nan]},
        index=pd.Index(range(2000, 2007), name="Year"),
    )
    out = dpl.fill_internal(df, fill="Spline")
    v = out["S"].to_numpy()
    assert np.isnan(v[0]) and np.isnan(v[-1])          # leading/trailing NaN kept
    assert not np.isnan(v[3])                          # interior gap filled
    assert list(np.round(v[[1, 2, 4, 5]], 6)) == [0.5, 0.6, 0.9, 1.0]  # data untouched


def test_fill_short_series_unchanged():
    df = pd.DataFrame({"S": [np.nan, 0.5, np.nan]},
                      index=pd.Index([1, 2, 3], name="Year"))
    out = dpl.fill_internal(df, fill="Mean")
    assert out["S"].isna().tolist() == [True, False, True]   # <=1 value: no fill


def test_fill_bad_method():
    with pytest.raises(ValueError):
        dpl.fill_internal(_frame(), fill="Bogus")
    with pytest.raises(TypeError):
        dpl.fill_internal([1, 2, 3], fill="Mean")


def _correlated_frame():
    # deterministic: a shared signal, one series carrying a growth trend
    yrs = np.arange(1900, 1950)
    t = np.arange(50)
    common = 1.0 + 0.2 * np.sin(t / 4.0)
    growth = np.linspace(1.5, 0.7, 50)
    df = pd.DataFrame(
        {"S1": growth * common, "S2": common, "S3": 1.01 * common},
        index=pd.Index(yrs, name="Year"),
    )
    return df, list(range(1920, 1926))          # a 6-year interior gap in S1


def test_fill_arstan_preserves_data_and_fills_gap():
    df, gap = _correlated_frame()
    truth = df["S1"].copy()
    df.loc[gap[0]:gap[-1], "S1"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.fill_internal(df, fill="ARSTAN")

    present = ~df["S1"].isna()
    assert np.allclose(out["S1"][present], df["S1"][present])       # real rings kept
    assert out[["S2", "S3"]].equals(df[["S2", "S3"]])               # others untouched
    filled = out["S1"].loc[gap[0]:gap[-1]]
    assert filled.notna().all()                                    # gap filled
    assert (filled > 0).all()                                      # positive widths
    # filled values track the (growth-modulated) truth reasonably
    assert np.abs(filled.to_numpy() - truth.loc[gap[0]:gap[-1]].to_numpy()).mean() < 0.15


def test_fill_arstan_leaves_leading_trailing_nan():
    df, _ = _correlated_frame()
    df.loc[1900:1902, "S2"] = np.nan                 # leading gap
    df.loc[1947:1949, "S3"] = np.nan                 # trailing gap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dpl.fill_internal(df, fill="ARSTAN")
    assert out["S2"].loc[1900:1902].isna().all()
    assert out["S3"].loc[1947:1949].isna().all()
