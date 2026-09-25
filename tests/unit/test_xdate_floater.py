import io
import contextlib

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _drop_and_recover(rwl_path, target, **kw):
    """Pull one dated series out of a collection, hand its bare ring widths back
    as a 'floating' series, and crossdate it against the remaining (dated) master.
    Returns (true_first, true_last, result)."""
    rwl = _quiet(dpl.readers, rwl_path)
    truth = rwl[target].dropna()
    reference = rwl.drop(columns=[target])
    res = _quiet(dpl.xdate_floater, reference, truth.to_numpy(),
                 series_name=target, verbose=False, **kw)
    return int(truth.index.min()), int(truth.index.max()), res


def test_recovers_known_date_ca533():
    first, last, res = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011")
    assert res["best"]["min_year"] == first
    assert res["best"]["max_year"] == last


def test_recovers_known_date_co021():
    first, last, res = _drop_and_recover("tests/data/rwl/co021.rwl", "641114")
    assert res["best"]["min_year"] == first
    assert res["best"]["max_year"] == last


def test_recovers_date_when_floater_longer_than_reference():
    # Floating the deepest core (CAM211) removes it from the reference, so the
    # remaining leave-one-out master is SHORTER than the floater (ny > nx) and the
    # master lies entirely inside the floater at the correct offset. This used to
    # raise an IndexError in the sliding-search crawl (a negative-index slice);
    # it must now date correctly like any other series.
    first, last, res = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM211")
    assert res["best"]["min_year"] == first
    assert res["best"]["max_year"] == last


def test_future_best_date_is_flagged():
    # dplPy (unlike dplR) scores placements past the reference into future years.
    # If the WINNER lands past the present calendar year the date is impossible and
    # must be flagged. Build a floater whose true best placement runs into the
    # future: its first 60 rings match the reference's most recent 60 years, then
    # 10 more rings extend past the present.
    import datetime
    this = datetime.date.today().year
    rng = np.random.default_rng(7)
    s = rng.random(120) + 0.5
    years = list(range(this - 119, this + 1))          # reference ends at present
    ref = pd.DataFrame({"R%d" % k: s + rng.normal(0, 1e-3, 120) for k in range(4)},
                       index=years)
    floater = np.concatenate([s[60:120], rng.random(10) + 0.5])
    res = _quiet(dpl.xdate_floater, ref, floater, series_name="FUT01",
                 min_overlap=50, verbose=False)
    assert res["best"]["max_year"] > this
    assert res["best"]["date_warning"] == "future"


def test_normal_best_date_has_no_future_warning():
    _, _, res = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011")
    assert res["best"]["date_warning"] is None


def test_fd_transform_recovers_date():
    first, last, res = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011",
                                         transform="fd")
    assert res["best"]["min_year"] == first
    assert res["best"]["max_year"] == last


def test_best_is_by_t_and_decisive():
    _, _, res = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011")
    stats = res["floater_cor_stats"]
    # table is sorted best-first by t, and the top match stands clearly above the rest
    assert stats["t"].iloc[0] == res["best"]["t"]
    assert stats["t"].iloc[0] - stats["t"].iloc[1] > 3.0
    assert (stats["n"] >= 50).all()             # every offset meets the minimum overlap
    span = stats["max_year"] - stats["min_year"] + 1
    assert (span == span.iloc[0]).all()         # full-timber span everywhere


def test_crossdating_stats_present_and_sane():
    _, _, res = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011")
    for col in ("r", "t", "eff_df", "p_bonf", "n"):
        assert col in res["floater_cor_stats"].columns
    b = res["best"]
    for key in ("t", "eff_df", "p_bonf", "one_over_p", "isolation_factor"):
        assert key in b
    # the true date is a strong, highly significant match
    assert b["t"] > 3.5                          # classic acceptance threshold
    assert b["p_bonf"] < 0.01
    assert b["isolation_factor"] > 10            # clearly isolated from the runner-up


def test_effective_df_reduced_for_autocorrelation():
    # prewhitened series carry little autocorrelation, so eff_df stays near n;
    # differenced series carry induced (negative) AC, so eff_df drops below n.
    _, _, pw = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011", transform="pw")
    _, _, fd = _drop_and_recover("tests/data/rwl/ca533.rwl", "CAM011", transform="fd")
    assert pw["best"]["eff_df"] <= pw["best"]["n"]
    assert fd["best"]["eff_df"] < fd["best"]["n"]        # differencing lowers eff df


def test_prewhiten_alias_and_transform_validation():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    truth = rwl["CAM011"].dropna().to_numpy()
    ref = rwl.drop(columns=["CAM011"])
    b_alias = _quiet(dpl.xdate_floater, ref, truth, prewhiten=True, verbose=False)["best"]
    b_pw = _quiet(dpl.xdate_floater, ref, truth, transform="pw", verbose=False)["best"]
    assert b_alias == b_pw
    with pytest.raises(ValueError):
        _quiet(dpl.xdate_floater, ref, truth, transform="bogus", verbose=False)


def test_accepts_series_list_and_dataframe():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    truth = rwl["CAM011"].dropna()
    ref = rwl.drop(columns=["CAM011"])
    b_arr = _quiet(dpl.xdate_floater, ref, truth.to_numpy(), verbose=False)["best"]
    b_list = _quiet(dpl.xdate_floater, ref, list(truth.to_numpy()), verbose=False)["best"]
    b_ser = _quiet(dpl.xdate_floater, ref, truth, verbose=False)["best"]
    b_df = _quiet(dpl.xdate_floater, ref, truth.to_frame(), verbose=False)["best"]
    assert b_arr == b_list == b_ser == b_df


def test_return_rwl_places_series():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    truth = rwl["CAM011"].dropna()
    ref = rwl.drop(columns=["CAM011"])
    res = _quiet(dpl.xdate_floater, ref, truth.to_numpy(), series_name="CAM011",
                 return_rwl=True, verbose=False)
    placed = res["placed"]
    assert list(placed.columns) == ["CAM011"]
    assert int(placed.index.min()) == res["best"]["min_year"]
    assert int(placed.index.max()) == res["best"]["max_year"]
    assert np.allclose(placed["CAM011"].to_numpy(), truth.to_numpy())
    assert "combined" in res


def test_min_overlap_validation():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    ref = rwl.drop(columns=["CAM011"])
    short = rwl["CAM011"].dropna().to_numpy()[:30]
    with pytest.raises(ValueError):
        _quiet(dpl.xdate_floater, ref, short, min_overlap=50, verbose=False)


def test_empty_series_raises():
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    with pytest.raises(ValueError):
        _quiet(dpl.xdate_floater, rwl, [np.nan, np.nan], verbose=False)


def test_make_plot_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rwl = _quiet(dpl.readers, "tests/data/rwl/ca533.rwl")
    truth = rwl["CAM011"].dropna().to_numpy()
    ref = rwl.drop(columns=["CAM011"])
    res = _quiet(dpl.xdate_floater, ref, truth, series_name="CAM011",
                 make_plot=True, verbose=False)
    assert "placed" in res            # make_plot also builds the placed series
    plt.close("all")


def test_exported():
    assert hasattr(dpl, "xdate_floater")
