import math

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl
from dplpy.sensitivity import _sens1_value, _sens2_value


# --- brute-force references, transcribed straight from dplR 1.7.9 src/sens.c --
def _ref_sens1(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return float("nan")
    s = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(1, n):
            prev, cur = x[i - 1], x[i]
            term = abs(cur - prev) / (cur + prev)
            if not math.isnan(term):      # dplR skips NaN terms only
                s += term
    return (s + s) / (n - 1)


def _ref_sens2(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return float("nan")
    sum1 = sum(abs(x[i] - x[i - 1]) for i in range(1, n))
    sum2 = sum(x)
    return sum1 / (sum2 - sum2 / n)


# --- hand-computed values -----------------------------------------------------
def test_sens1_hand_value_trend():
    # [1,2,3,4]: 2*(1/3 + 1/5 + 1/7)/3
    expected = 2 * (1 / 3 + 1 / 5 + 1 / 7) / 3
    assert _sens1_value([1, 2, 3, 4]) == pytest.approx(expected, rel=1e-15)
    assert dpl.sens1(pd.Series([1, 2, 3, 4])) == pytest.approx(expected, rel=1e-15)


def test_sens2_hand_value_trend():
    # [1,2,3,4]: sum|diff|=3, sum=10, 3/(10 - 10/4) = 3/7.5 = 0.4
    assert _sens2_value([1, 2, 3, 4]) == pytest.approx(0.4, rel=1e-15)


def test_sens1_equals_sens2_when_no_trend():
    # symmetric zig-zag about a constant mean -> the two agree
    x = [1, 2, 1, 2, 1, 2]
    assert _sens1_value(x) == pytest.approx(_sens2_value(x), rel=1e-12)


# --- edge cases dplR encodes --------------------------------------------------
def test_sens1_skips_zero_over_zero_pair():
    # (0,0) -> 0/0 term skipped, but the divisor stays n-1 = 3
    # remaining terms: (0,1)->1, (1,2)->1/3 ; sens1 = 2*(1 + 1/3)/3
    expected = 2 * (1 + 1 / 3) / 3
    assert _sens1_value([0, 0, 1, 2]) == pytest.approx(expected, rel=1e-15)
    assert _sens1_value([0, 0, 1, 2]) == pytest.approx(_ref_sens1([0, 0, 1, 2]), rel=1e-15)


def test_fewer_than_two_values_is_nan():
    assert math.isnan(_sens1_value([5.0]))
    assert math.isnan(_sens2_value([5.0]))
    assert math.isnan(_sens1_value([]))
    # one non-NA value after dropping NaNs -> nan
    assert math.isnan(dpl.sens1(pd.Series([np.nan, np.nan, 3.0])))


def test_na_values_are_dropped():
    # NaNs removed first, so these two are identical series
    a = _sens1_value([1, 2, np.nan, 3, 4])
    b = _sens1_value([1, 2, 3, 4])
    assert a == pytest.approx(b, rel=1e-15)


# --- DataFrame / Series dispatch ----------------------------------------------
def test_dataframe_returns_series_per_column():
    df = pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [2.0, 2, 2, 2]})
    out = dpl.sens1(df)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]
    assert out.name == "sens1"
    assert out["A"] == pytest.approx(_sens1_value(df["A"].to_numpy()), rel=1e-15)
    assert out["B"] == pytest.approx(0.0)          # constant series -> zero sensitivity


def test_series_input_matches_column_value():
    df = pd.DataFrame({"A": [1.0, 3, 2, 5, 4], "B": [1.0, 1, 2, 3, 5]})
    assert dpl.sens2(df["A"]) == pytest.approx(dpl.sens2(df)["A"], rel=1e-15)


# --- machine-precision agreement with the reference on real data --------------
def test_matches_reference_on_ca533():
    data = dpl.readers("./tests/data/csv/ca533.csv")
    s1 = dpl.sens1(data)
    s2 = dpl.sens2(data)
    for col in data.columns:
        x = data[col].to_numpy()
        assert s1[col] == pytest.approx(_ref_sens1(x), rel=1e-12, nan_ok=True)
        assert s2[col] == pytest.approx(_ref_sens2(x), rel=1e-12, nan_ok=True)
    # sanity: ca533 mean sensitivities are small positive numbers
    assert (s1.dropna() > 0).all() and (s1.dropna() < 1).all()
