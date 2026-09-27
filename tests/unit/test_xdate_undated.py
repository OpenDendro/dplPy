import io
import contextlib

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _synthetic_floats(n_series=6, seg_len=200, step=55, noise=0.06, seed=0):
    """Cut overlapping windows from the ca533 site mean (a real, autocorrelated
    signal) at KNOWN staggered offsets, add small per-series noise, and strip the
    dates. Returns (undated_collection, true_start_offsets)."""
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    base = rwl.mean(axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    cols, true = {}, {}
    for k in range(n_series):
        start = k * step
        seg = base[start:start + seg_len].copy()
        seg = seg + rng.normal(0, noise * np.nanstd(seg), size=len(seg))
        cols["T%02d" % k] = pd.Series(seg)
        true["T%02d" % k] = start
    return pd.DataFrame(cols), true


def test_exported():
    assert hasattr(dpl, "xdate_undated")


def test_recovers_relative_offsets():
    # Overlapping windows cut at known offsets must be reassembled with the correct
    # RELATIVE spacing (up to one global shift) for every confidently-placed series.
    und, true = _synthetic_floats(seed=1)
    res = _quiet(dpl.xdate_undated, und)
    off = res["offsets"]
    v = res["verify"].set_index("series")
    conf = [nm for nm in off.index if not bool(v.loc[nm, "review"])]
    assert len(conf) >= 5                          # nearly all place confidently
    rec = off.loc[conf, "start_year"]
    tru = pd.Series({nm: true[nm] for nm in conf})
    resid = (rec - rec.mean()) - (tru - tru.mean())
    assert resid.abs().max() == 0                  # exact relative recovery


def test_anchor_year_places_earliest_at_one():
    und, _ = _synthetic_floats(seed=1)
    res = _quiet(dpl.xdate_undated, und)
    assert int(res["offsets"]["start_year"].min()) == 1        # default anchor
    res2 = _quiet(dpl.xdate_undated, und, anchor_year=1000)
    assert int(res2["offsets"]["start_year"].min()) == 1000    # custom anchor


def test_outputs_structure():
    und, _ = _synthetic_floats(seed=2)
    res = _quiet(dpl.xdate_undated, und)
    for key in ("pairwise", "offsets", "placed", "chronology", "order", "verify",
                "unplaced"):
        assert key in res
    assert list(res["pairwise"].columns) == ["series_i", "series_j", "lag", "n", "r", "t"]
    assert list(res["verify"].columns) == \
        ["series", "loo_best_pos", "loo_t", "isolation", "agrees", "review"]
    assert isinstance(res["chronology"], pd.Series)
    # the chronology spans exactly the placed extent, anchored at 1
    assert int(res["chronology"].index.min()) == 1
    assert int(res["chronology"].index.max()) == int(res["offsets"]["end_year"].max())


def test_pairwise_symmetric_lag_sign():
    # A pair's recorded lag is the offset of series_j relative to series_i; for windows
    # cut at known offsets it must equal the true spacing (up to the global anchor).
    und, true = _synthetic_floats(n_series=3, seed=5)
    res = _quiet(dpl.xdate_undated, und)
    pw = res["pairwise"]
    row = pw.iloc[0]                               # strongest pair
    expected = true[row["series_j"]] - true[row["series_i"]]
    assert row["lag"] == expected


def test_min_series_raises():
    with pytest.raises(ValueError):
        _quiet(dpl.xdate_undated, pd.DataFrame({"only": [1.0, 2.0, 3.0]}))


def test_ar_order_params_accepted():
    # The AR-order controls are exposed and both the COFECHA-preset default and the
    # dplR global-AIC setting produce a valid assembled chronology.
    und, _ = _synthetic_floats(seed=1)
    res = _quiet(dpl.xdate_undated, und, ar_max=None, first_aic_min=False)
    assert isinstance(res["chronology"], pd.Series)
    assert int(res["offsets"]["start_year"].min()) == 1


def test_make_plot_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    und, _ = _synthetic_floats(seed=3)
    res = _quiet(dpl.xdate_undated, und, make_plot=True)
    assert "chronology" in res
    plt.close("all")
