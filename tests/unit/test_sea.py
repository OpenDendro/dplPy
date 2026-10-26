import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl
from dplpy.sea import _block_shuffle_row


# Synthetic AR(1) series with injected cooling at known events, regenerated
# deterministically (np.random.default_rng is a stable stream). This is the series
# the SEA golden master was built on; see dev/sea/.
def _synthetic():
    rng = np.random.default_rng(20260501)
    yr = np.arange(1000, 2001)
    n = len(yr)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + e[t]
    x = (x - x.mean()) / x.std()
    events = np.array([1104, 1258, 1453, 1601, 1641, 1695, 1783, 1809,
                       1835, 1884, 1920, 1963])
    decay = np.array([2.5, 2.0, 1.3, 0.7, 0.3])
    y = x.copy()
    for ev in events:
        i = np.where(yr == ev)[0][0]
        for k, d in enumerate(decay):
            if i + k < n:
                y[i + k] -= d
    return pd.Series(y, index=yr), events


# Composite (sval0) from the MATLAB sea_dbl golden master, pre=5 post=10.
_GM_COMPOSITE = np.array([
    0.096374, 0.141989, -0.04309, -0.014598, -0.180676, -2.704397, -2.277166,
    -1.271243, -0.216029, 0.162254, -0.162378, -0.403599, -0.307787, -0.214887,
    -0.141796, 0.188117])
_GM_R1 = 0.566064
_GM_P = 3


def test_sea_composite_matches_matlab_golden_master():
    ser, events = _synthetic()
    res = dpl.sea(ser, events, pre=5, post=10, double_bootstrap=False,
                  boot_method="block", n_boot=10, seed=1)
    assert np.allclose(res.composite, _GM_COMPOSITE, atol=1e-5)
    assert abs(res.r1 - _GM_R1) < 1e-5
    assert res.block_length == _GM_P
    assert list(res.lags) == list(range(-5, 11))


def test_block_shuffle_kernel_fixed_example():
    row = np.arange(1.0, 17.0)
    oddcase = np.arange(0, 16, 3)
    out = _block_shuffle_row(row, 3, 16 % 3, oddcase, 2,
                             np.array([3, 0, 4, 1, 2]), 5)
    expected = np.array([2, 3, 8, 9, 10, 4, 5, 6, 11, 12, 13, 7, 14, 15, 16, 1.0])
    assert np.array_equal(out, expected)
    # value-preserving: a reshuffle of the same multiset
    assert np.array_equal(np.sort(out), np.sort(row))


def test_sea_double_bootstrap_envelope():
    ser, events = _synthetic()
    res = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=500,
                  n_boot=200, boot_method="block", seed=3)
    assert res.envelope is not None and res.envelope.shape == (3, 16)
    lo, med, hi = res.envelope
    assert np.all(lo <= med) and np.all(med <= hi)        # 5 <= 50 <= 95
    # composite is the all-event mean, unchanged by the bootstrap
    assert np.allclose(res.composite, _GM_COMPOSITE, atol=1e-5)


def test_sea_reproducible_with_seed():
    ser, events = _synthetic()
    a = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=300, seed=5)
    b = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=300, seed=5)
    assert np.array_equal(a.envelope, b.envelope)
    assert np.array_equal(a.null.to_numpy(), b.null.to_numpy())


def test_sea_composite_is_seed_independent():
    ser, events = _synthetic()
    a = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=300, seed=5)
    b = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=300, seed=99)
    assert np.allclose(a.composite, b.composite)


def test_sea_fallback_when_too_few_subsets_warns():
    ser, events = _synthetic()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=1000, seed=1)
    assert res.envelope is None                            # C(12,6)=924 < 1000
    assert any("ordinary SEA" in str(x.message) for x in w)


def test_sea_excises_out_of_range_events_with_warning():
    ser, events = _synthetic()
    events2 = np.append(events, 1998)                      # window would run past 2000
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = dpl.sea(ser, events2, pre=5, post=10, double_bootstrap=False, seed=1)
    assert 1998 not in res.events
    assert any("excising" in str(x.message) for x in w)


def test_sea_bad_boot_method_raises():
    ser, events = _synthetic()
    with pytest.raises(ValueError):
        dpl.sea(ser, events, boot_method="bogus")


def test_sea_accepts_series_or_pair():
    ser, events = _synthetic()
    a = dpl.sea(ser, events, pre=5, post=10, double_bootstrap=False, seed=1)
    b = dpl.sea((ser.to_numpy(), ser.index.to_numpy()), events, pre=5, post=10,
                double_bootstrap=False, seed=1)
    assert np.allclose(a.composite, b.composite)


def test_sea_random_null_runs():
    ser, events = _synthetic()
    res = dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=300,
                  n_boot=200, boot_method="random", seed=1)
    assert res.null.shape == (6, 16)
    assert list(res.null.index) == [1, 5, 10, 90, 95, 99]
    assert res.r1 is None                                  # no block length for random


def test_sea_does_not_mutate_input():
    ser, events = _synthetic()
    before = ser.copy(deep=True)
    dpl.sea(ser, events, pre=5, post=10, n_sample=6, n_subsets=300, seed=1)
    pd.testing.assert_series_equal(ser, before)
